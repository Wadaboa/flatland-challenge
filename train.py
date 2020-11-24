import os
import copy
import random
import copy

from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from action_selectors import EpsilonGreedyActionSelector
from env_utils import RailEnvChoices
from predictions import ShortestPathPredictor, NullPredictor
from binary_tree_obs import BinaryTreeObservator
from graph_obs import GraphObservator
from policies import DQNPolicy, DQNGNNPolicy
from environments import RailEnvWrapper
import utils
import env_utils


RANDOM_SEED = 1
WRITER = SummaryWriter()
OBSERVATORS = {
    "tree": TreeObsForRailEnv,
    "binary_tree": BinaryTreeObservator,
    "graph": GraphObservator
}
PREDICTORS = {
    "tree": ShortestPathPredictorForRailEnv,
    "binary_tree": ShortestPathPredictor,
    "graph": ShortestPathPredictor
}
POLICIES = {
    "tree": DQNPolicy,
    "binary_tree": DQNPolicy,
    "graph": DQNGNNPolicy,
}


def set_num_threads(num_threads):
    '''
    Set the maximum number of threads PyTorch can use
    '''
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


def tensorboard_log(name, x, y, plot=['min', 'max', 'mean', 'std', 'hist']):
    '''
    Log the given x/y values to tensorboard
    '''
    if not isinstance(x, np.ndarray) and not isinstance(x, list):
        WRITER.add_scalar(name, x, y)
    else:
        if ((isinstance(x, list) and len(x) == 0) or
                (isinstance(x, np.ndarray) and x.size == 0)):
            return
        if 'min' in plot:
            WRITER.add_scalar(f"{name}_min", np.min(x), y)
        if 'max' in plot:
            WRITER.add_scalar(f"{name}_max", np.max(x), y)
        if 'mean' in plot:
            WRITER.add_scalar(f"{name}_mean", np.mean(x), y)
        if 'std' in plot:
            WRITER.add_scalar(f"{name}_std", np.std(x), y)
        if 'hist' in plot:
            WRITER.add_histogram(name, np.array(x), y)


def format_choices_probabilities(choices_probabilities):
    '''
    Helper function to pretty print choices probabilities
    '''
    choices_probabilities = np.round(choices_probabilities, 3)
    choices = ["←", "→", "◼"]

    buffer = ""
    for choice, choice_prob in zip(choices, choices_probabilities):
        buffer += choice + " " + "{:^4.2%}".format(choice_prob) + " "

    return buffer


def create_rail_env(args, env=""):
    '''
    Build a RailEnv object with the specified parameters
    '''
    # Check if an environment file is provided
    if env:
        rail_generator = rail_from_file(env)
    else:
        rail_generator = sparse_rail_generator(
            max_num_cities=args.max_cities,
            grid_mode=args.grid,
            max_rails_between_cities=args.max_rails_between_cities,
            max_rails_in_city=args.max_rails_in_cities,
            seed=RANDOM_SEED
        )
    predictor = PREDICTORS[args.observation](max_depth=args.max_depth)
    observator = OBSERVATORS[args.observation](args.max_depth, predictor)
    # Initialize malfunctions
    malfunctions = None
    if args.malfunction:
        malfunctions = ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=args.malfunction_rate,
                min_duration=args.malfunction_min_duration,
                max_duration=args.malfunction_max_duration
            )
        )

    # Initialize agents speeds
    speed_map = None
    if args.variable_speed:
        speed_map = {
            1.: 0.25,
            1. / 2.: 0.25,
            1. / 3.: 0.25,
            1. / 4.: 0.25
        }
    schedule_generator = sparse_schedule_generator(speed_map, seed=RANDOM_SEED)

    # Build the environment
    return RailEnvWrapper(
        width=args.width,
        height=args.height,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=args.num_trains,
        obs_builder_object=observator,
        malfunction_generator=malfunctions,
        remove_agents_at_target=True,
        random_seed=RANDOM_SEED
    )


def copy_obs(obs):
    if hasattr(obs, "copy"):
        return obs.copy()
    return copy.deepcopy(obs)


def train_agents(args):
    '''
    Train and evaluate agents on the specified environments
    '''
    # Initialize threads and seeds
    set_num_threads(args.num_threads)
    if args.fix_random:
        utils.fix_random(RANDOM_SEED)

    # Setup the environments
    train_env = create_rail_env(args, env=args.train_env)
    val_env = create_rail_env(args, env=args.val_env)

    # Define "static" random seeds for evaluation purposes
    val_seeds = [
        env_utils.get_seed(val_env)
        for e in range(args.n_val_episodes)
    ]

    # Set state size and action size
    choice_size = 3
    avg_score = 0.0
    avg_completion = 0.0

    # Initialize the agents policy
    policy = POLICIES[args.observation](
        train_env.state_size, choice_size,
        EpsilonGreedyActionSelector(
            epsilon_start=args.eps_start, epsilon_decay=args.eps_decay, epsilon_end=args.eps_end
        ), training=True
    )

    # Handle replay buffer
    if args.restore_replay_buffer:
        try:
            policy.load_replay_buffer(args.restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print(
                "\n🛑 Could't load replay buffer, were the experiences generated using the same depth?"
            )
            print(e)
            exit(1)
    print("\n💾 Replay buffer status: {}/{} experiences".format(
        len(policy.memory), policy.PARAMETERS["buffer_size"]
    ))

    # Set the unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%Y%m%d-%H%M%S')

    # Print initial training info
    training_timer = utils.Timer()
    training_timer.start()
    print("\n🚉 Starting training \t Training {} trains on {}x{} grid for {} episodes \tEvaluating on {} episodes every {} episodes".format(
        args.num_trains,
        args.width, args.height,
        args.n_train_episodes,
        args.n_val_episodes,
        args.checkpoint
    ))
    print(f"\n🧠 Model with training id {training_id}\n")

    # Do the specified number of episodes
    for episode in range(args.n_train_episodes + 1):
        # Initialize timers
        step_timer = utils.Timer()
        reset_timer = utils.Timer()
        learn_timer = utils.Timer()
        inference_timer = utils.Timer()

        # Reset environment and renderer
        reset_timer.start()
        if args.fix_random:
            obs, info = train_env.reset(random_seed=RANDOM_SEED)
        else:
            obs, info = train_env.reset(regenerate_rail=True, regenerate_schedule=True,
                                        activate_agents=False, random_seed=env_utils.get_seed(train_env))
        reset_timer.end()
        if args.render_every and not args.render_val and episode % args.render_every == 0:
            env_renderer = RenderTool(
                train_env,
                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                show_debug=True,
                screen_height=1080,
                screen_width=1920
            )

        # Compute agents with same source
        agents_with_same_start = env_utils.get_agents_same_start(train_env)

        # Initialize data structures for training info
        score, steps = 0, 0
        choices_taken = []
        legal_choices = dict()
        update_values = [False] * args.num_trains
        choices_count = [0] * choice_size
        action_dict = dict()
        choice_dict = dict()
        rewards = dict()
        prev_obs = dict()
        prev_choices = dict()
        arrived_agents = set()
        for handle in range(args.num_trains):
            legal_choices[handle] = train_env.railway_encoding.get_legal_choices(
                handle,
                train_env.railway_encoding.get_agent_actions(handle)
            )
            rewards[handle] = 0
            prev_choices[handle] = RailEnvChoices.CHOICE_LEFT.value
            if obs[handle] is not None:
                prev_obs[handle] = copy_obs(obs[handle])

        # Do an episode
        for step in range(args.max_moves):
            # Inference step
            inference_timer.start()

            # Prioritize entry of faster agents in the environment
            for position in agents_with_same_start:
                if len(agents_with_same_start[position]) > 0:
                    del agents_with_same_start[position][0]
                    for agent in agents_with_same_start[position]:
                        info['action_required'][agent] = False

            for agent in train_env.get_agent_handles():
                update_values[agent] = False
                if info['action_required'][agent]:
                    if train_env.railway_encoding.is_real_decision(agent):
                        legal_actions = train_env.railway_encoding.get_agent_actions(
                            agent)
                        legal_choices[agent] = train_env.railway_encoding.get_legal_choices(
                            agent, legal_actions
                        )
                        choice = policy.act(obs[agent], legal_choices[agent])
                        action = train_env.railway_encoding.map_choice_to_action(
                            choice, legal_actions
                        )
                        assert action != RailEnvActions.DO_NOTHING.value, (
                            choice, legal_actions)
                        update_values[agent] = True
                        choices_count[choice] += 1
                        choices_taken.append(choice)
                        choice_dict.update({agent: choice})
                    else:
                        actions = train_env.railway_encoding.get_agent_actions(
                            agent)
                        assert len(actions) == 1, actions
                        action = actions[0]
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if it's currently malfunctioning,
                    # or if it's in deadlock or if it's in the middle of traversing a cell
                    action = RailEnvActions.DO_NOTHING.value
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if args.render_every and not args.render_val and episode % args.render_every == 0:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=True, show_rowcols=True
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():

                # Accumulate rewards for choices if the agent is not arrived,
                # otherwise return a positive reward equal to the maximum
                # number of steps
                if done[agent]:
                    rewards[agent] = args.max_moves
                else:
                    rewards[agent] += all_rewards[agent]

                # Only learn from timesteps where something happened
                if update_values[agent] or (done[agent] and agent not in arrived_agents):
                    learn_timer.start()
                    experience = (
                        prev_obs[agent], prev_choices[agent], rewards[agent], obs[agent], legal_choices[agent], done[agent]
                    )
                    policy.step(experience)
                    rewards[agent] = 0
                    learn_timer.end()
                    prev_obs[agent] = copy_obs(obs[agent])
                    prev_choices[agent] = choice_dict[agent]

                # Add agent to the list of arrived agents
                if done[agent]:
                    arrived_agents.add(agent)

                # Update observation and score
                if next_obs[agent] is not None:
                    obs[agent] = copy_obs(next_obs[agent])
                score += all_rewards[agent]

            # Break if every agent arrived
            steps = step
            if done['__all__'] or env_utils.check_if_all_blocked(train_env):
                break

        # Close window
        if args.render_every and not args.render_val and episode % args.render_every == 0:
            env_renderer.close_window()

        # Epsilon decay
        policy.choice_selector.decay()

        # Save final scores
        tasks_finished = sum(
            done[i] for i in train_env.get_agent_handles()
        )
        completion = tasks_finished / train_env.get_num_agents()
        normalized_score = (
            score / (args.max_moves * train_env.get_num_agents())
        )
        avg_completion = (
            episode * avg_completion + completion
        ) / (episode + 1)
        avg_score = (episode * avg_score + normalized_score) / (episode + 1)
        choices_probs = choices_count / np.sum(choices_count)

        # Save model and replay buffer at checkpoint
        if episode % args.checkpoint == 0:
            policy.save(
                './checkpoints/' + training_id + '-' + str(episode)
            )
            if args.save_replay_buffer:
                policy.save_replay_buffer(
                    './replay_buffers/' + training_id +
                    '-' + str(episode) + '.pkl'
                )

        # Print episode info
        print(
            '\r🚂 Episode {:4n}'
            '\t 🏆 Score: {:<+4.3f}'
            ' Avg: {:>+4.3f}'
            '\t 💯 Done: {:<7.2%}'
            ' Avg: {:>7.2%}'
            '\t 🦶 Steps {:3n}'
            '\t 🎲 Epsilon: {:4.3f} '
            '\t 🔀 Choices probabilities: {:^}'.format(
                episode,
                normalized_score,
                avg_score,
                completion,
                avg_completion,
                steps,
                policy.choice_selector.epsilon,
                format_choices_probabilities(choices_probs)
            ), end="\n"
        )

        # Evaluate policy and log results at some interval
        if episode > 0 and episode % args.checkpoint == 0 and args.n_val_episodes > 0:
            scores, completions, val_steps = eval_policy(
                args, val_env, policy, val_seeds
            )

            # Log validation metrics to tensorboard
            tensorboard_log(
                "validation/scores", scores, episode,
                plot=['mean', 'std', 'hist']
            )
            tensorboard_log(
                "validation/completions", completions, episode,
                plot=['mean', 'std', 'hist']
            )
            tensorboard_log(
                "validation/steps", val_steps, episode,
                plot=['mean', 'std', 'hist']
            )

        # Log training actions info to tensorboard
        tensorboard_log(
            "choices/left", choices_probs[env_utils.RailEnvChoices.CHOICE_LEFT.value], episode
        )
        tensorboard_log(
            "choices/right", choices_probs[env_utils.RailEnvChoices.CHOICE_RIGHT.value], episode
        )
        tensorboard_log(
            "choices/stop", choices_probs[env_utils.RailEnvChoices.STOP.value], episode
        )

        # Log training info to tensorboard
        tensorboard_log("training/steps", steps, episode)
        tensorboard_log(
            "training/epsilon", policy.choice_selector.epsilon, episode
        )
        tensorboard_log(
            "training/loss", policy.loss.data.item(), episode
        )
        tensorboard_log("training/score", normalized_score, episode)
        tensorboard_log("training/completion", completion, episode)
        tensorboard_log(
            "training/buffer_size", len(policy.memory), episode
        )

        # Log training time info to tensorboard
        tensorboard_log("timer/reset", reset_timer.get(), episode)
        tensorboard_log("timer/step", step_timer.get(), episode)
        tensorboard_log("timer/learn", learn_timer.get(), episode)
        tensorboard_log(
            "timer/total", training_timer.get_current(), episode
        )

    print("\n\r🏁 Training Ended \tTrained {} trains on {}x{} grid for {} episodes \t Evaluated on {} episodes every {} episodes".format(
        args.num_trains,
        args.width, args.height,
        args.n_train_episodes,
        args.n_val_episodes,
        args.checkpoint
    ))
    print("\n💾 Replay buffer status: {}/{} experiences".format(
        len(policy.memory), policy.PARAMETERS["buffer_size"]
    ))

    print(f"\n🧠 Saving model with training id {training_id}")
    policy.save('./checkpoints/' + training_id + '-latest')
    if args.save_replay_buffer:
        policy.save_replay_buffer(
            './replay_buffers/' + training_id + '-latest' + '.pkl'
        )

    # Close Tensorboard writer
    WRITER.close()


def eval_policy(args, env, policy, val_seeds):
    '''
    Perform a validation round with the given policy
    in the specified environment
    '''
    action_dict = dict()
    scores, completions, steps = [], [], []

    # Do the specified number of episodes
    print('\nStarting validation:')
    for episode, seed in enumerate(val_seeds):
        score = 0.0
        final_step = 0
        obs, info = env.reset(
            regenerate_rail=True, regenerate_schedule=True,
            random_seed=seed
        )
        if args.render_every and args.render_val and episode % args.render_every == 0:
            env_renderer = RenderTool(
                env,
                agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                show_debug=True,
                screen_height=1080,
                screen_width=1920
            )

        # Compute agents with same source
        agents_with_same_start = env_utils.get_agents_same_start(env)

        # Do an episode
        for step in range(args.max_moves):

            # Prioritize enter of faster agent in the environment
            for position in agents_with_same_start:
                if len(agents_with_same_start[position]) > 0:
                    del agents_with_same_start[position][0]
                    for agent in agents_with_same_start[position]:
                        info['action_required'][agent] = False

            # Perform a step
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    if env.railway_encoding.is_real_decision(agent):
                        legal_actions = env.railway_encoding.get_agent_actions(
                            agent)
                        legal_choices = env.railway_encoding.get_legal_choices(
                            agent, legal_actions
                        )
                        choice = policy.act(obs[agent], legal_choices)
                        action = env.railway_encoding.map_choice_to_action(
                            choice, legal_actions
                        )
                        assert action != RailEnvActions.DO_NOTHING.value, (
                            choice, legal_actions)
                    else:
                        actions = env.railway_encoding.get_agent_actions(agent)
                        assert len(actions) == 1, actions
                        action = actions[0]
                else:
                    action = RailEnvActions.DO_NOTHING.value
                action_dict.update({agent: action})

            # Perform a step in the environment
            obs, all_rewards, done, info = env.step(action_dict)

            # Render an episode at some interval
            if args.render_every and args.render_val and episode % args.render_every == 0:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=True, show_rowcols=True
                )

            # Update rewards
            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            # Break if every agent arrived
            final_step = step
            if done['__all__']:
                break

        # Close window
        if args.render_every and args.render_val and episode % args.render_every == 0:
            env_renderer.close_window()

        # Save final scores
        normalized_score = (
            score / (args.max_moves * env.get_num_agents())
        )
        scores.append(normalized_score)
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / env.get_num_agents()
        completions.append(completion)
        steps.append(final_step)

        print(
            '\r🚂 Validation {:3n}'
            '\t 🏆 Score: {:+4.3f}'
            '\t 💯 Done: {:7.2%}'
            '\t 🦶 Steps: {:4n}'.format(
                episode,
                normalized_score,
                completion,
                final_step
            ), end="\n"
        )

        # Log validation metrics to tensorboard
        tensorboard_log(
            "validation/env_scores", score, episode,
        )
        tensorboard_log(
            "validation/env_completions", completion, episode,
        )
        tensorboard_log(
            "validation/env_steps", final_step, episode
        )

    # Print validation results
    print(
        '\r✅ Validation End'
        '\t 🏆 Avg score: {:+1.3f}'
        '\t 💯 Avg done: {:7.1%}%'
        '\t 🦶 Avg steps: {:5.2f}'.format(
            np.mean(scores),
            np.mean(completions),
            np.mean(steps)
        ), end="\n\n"
    )
    return scores, completions, steps


def parse_args():
    '''
    Parse terminal arguments
    '''
    parser = ArgumentParser()

    # Environment parameters
    parser.add_argument(
        "--train_env", action='store', default="",
        help="path to the train environment file", type=str
    )
    parser.add_argument(
        "--val_env", action='store', default="",
        help="path to the validation environment file", type=str
    )
    parser.add_argument(
        "--num_trains", action='store', default=1,
        help="number of trains to spawn", type=int
    )
    parser.add_argument(
        "--width", action='store', default=48,
        help="environment width", type=int
    )
    parser.add_argument(
        "--height", action='store', default=27,
        help="environment height", type=int
    )
    parser.add_argument(
        "--malfunction", action='store_true', default=False,
        help="enable malfunctions"
    )
    parser.add_argument(
        "--malfunction_rate", action='store', default=0.2,
        help="malfunction rate", type=float
    )
    parser.add_argument(
        "--malfunction_min_duration", action='store', default=15,
        help="malfunction minimum duration", type=int
    )
    parser.add_argument(
        "--malfunction_max_duration", action='store', default=50,
        help="malfunction maximum duration", type=int
    )
    parser.add_argument(
        "--max_moves", action='store', default=300,  # 500
        help="maximum number of moves in an episode", type=int
    )
    parser.add_argument(
        "--max_depth", action='store', default=5,
        help="predictor maximum depth", type=int
    )
    parser.add_argument(
        "--max_cities", action='store', default=5,
        help="maximum number of cities where agents can start or end", type=int
    )
    parser.add_argument(
        "--grid", action='store_true',
        help="type of city distribution"
    )
    parser.add_argument(
        "--max_rails_between_cities", action='store', default=2,
        help="maximum number of tracks allowed between cities", type=int
    )
    parser.add_argument(
        "--max_rails_in_cities", action='store', default=3,
        help="maximum number of parallel tracks within a city", type=int
    )
    parser.add_argument(
        "--variable_speed", action='store_true', default=False,
        help="enable variable speed"
    )
    parser.add_argument(
        "--observation", action='store', default='graph', choices=OBSERVATORS.keys(),
        help="select observation type"
    )

    # Training parameters
    parser.add_argument(
        "--n_train_episodes", action='store', default=5000,
        help="number of episodes to run", type=int
    )
    parser.add_argument(
        "--n_val_episodes", action='store', default=25,
        help="number of evaluation episodes", type=int
    )
    parser.add_argument(
        "--checkpoint", action='store', default=100,
        help="checkpoint interval", type=int
    )
    parser.add_argument(
        "--eps_start", action='store', default=1.0,
        help="initial exploration", type=float
    )
    parser.add_argument(
        "--eps_end", action='store', default=0.01,
        help="final exploration", type=float
    )
    parser.add_argument(
        "--eps_decay", action='store', default=2e-4,  # 0.99
        help="exploration decay", type=float
    )
    parser.add_argument(
        "--restore_replay_buffer", action='store', default="",
        help="replay buffer to restore", type=str
    )
    parser.add_argument(
        "--save_replay_buffer", action='store_true',
        help="save replay buffer at each evaluation interval"
    )
    parser.add_argument(
        "--num_threads", action='store', default=1,
        help="number of threads PyTorch can use", type=int
    )
    parser.add_argument(
        "--render_every", action='store', default=150,
        help="how often to render an episode", type=int
    )
    parser.add_argument(
        "--render_val", action='store_true',
        help="render validation or training episodes"
    )
    parser.add_argument(
        "--fix_random", action='store_true',
        help="fix all the possible sources of randomness"
    )

    return parser.parse_args()


def main():
    '''
    Train environment with custom observation and prediction
    '''
    args = parse_args()
    train_agents(args)


if __name__ == "__main__":
    main()
