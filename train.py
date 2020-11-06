import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from predictions import ShortestPathPredictor
from observations import CustomObservation
from models import DDDQNPolicy
import utils


RANDOM_SEED = 1


def set_num_threads(num_threads):
    '''
    Set the maximum number of threads PyTorch can use
    '''
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


def tensorboard_log(writer, name, x, y):
    '''
    Log the given x/y values to tensorboard
    '''
    if type(x) in (int, float, str):
        writer.add_scalar(name, x, y)
    else:
        writer.add_scalar(f"{name}_min", np.min(x), y)
        writer.add_scalar(f"{name}_max", np.max(x), y)
        writer.add_scalar(f"{name}_mean", np.mean(x), y)
        writer.add_scalar(f"{name}_std", np.std(x), y)
        writer.add_histogram(name, np.array(x), y)


def format_action_probabilities(action_probabilities):
    '''
    Helper function to pretty print action probabilities
    '''
    action_probabilities = np.round(action_probabilities, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probabilities):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def create_rail_env(args, observation_builder, env=""):
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

    # Initialize malfunctions
    malfunctions = None
    if args.malfunction:
        malfunctions = ParamMalfunctionGen(MalfunctionParameters(
            malfunction_rate=args.malfunction_rate,
            min_duration=args.malfunction_min_duration,
            max_duration=args.malfunction_max_duration
        ))

    # Initialize agents speeds
    if args.variable_speed:
        speed_map = {
            1.: 0.25,
            1. / 2.: 0.25,
            1. / 3.: 0.25,
            1. / 4.: 0.25
        }
        schedule_generator = sparse_schedule_generator(speed_map)

    # Build the environment
    return RailEnv(
        width=args.width,
        height=args.height,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator if args.variable_speed else None,
        number_of_agents=args.num_trains,
        obs_builder_object=observation_builder,
        malfunction_generator=malfunctions,
        remove_agents_at_target=True,
        random_seed=RANDOM_SEED
    )


def train_agents(args):
    '''
    Train and evaluate agents on the specified environments
    '''
    # Initialize threads and seeds
    set_num_threads(args.num_threads)
    if args.fix_random:
        utils.fix_random(RANDOM_SEED)

    # Initialize predictor and observer
    predictor = ShortestPathPredictor(max_depth=args.max_depth)
    observation_builder = CustomObservation(
        max_depth=args.max_depth, predictor=predictor
    )

    # Setup the environments
    train_env = create_rail_env(args, observation_builder, env=args.train_env)
    val_env = create_rail_env(args, observation_builder, env=args.val_env)

    # Setup renderer
    if args.render:
        env_renderer = RenderTool(
            train_env,
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            screen_height=600,
            screen_width=800
        )

    # Set state size and action size
    state_size = (args.max_depth ** 2) * observation_builder.FEATURES
    action_size = 5

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_val_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_val_completion = 0.0
    train_smoothing = 0.99
    val_smoothing = 0.9

    # Initialize the agents policy
    policy = DDDQNPolicy(state_size, action_size, args)

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
        len(policy.memory.memory), args.buffer_size
    ))

    # Set tensorboard writer
    writer = SummaryWriter()

    # Set the unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%Y%m%d-%H%M%S')

    # Print initial training info
    training_timer = utils.Timer()
    training_timer.start()
    print("\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
        train_env.get_num_agents(),
        args.width, args.height,
        args.n_train_episodes,
        args.n_val_episodes,
        args.checkpoint_interval,
        training_id
    ))

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
            obs, info = train_env.reset(
                regenerate_rail=True, regenerate_schedule=True,
                random_seed=utils.get_seed(train_env)
            )
        reset_timer.end()
        if args.render:
            env_renderer.set_new_rail()

        # Initialize data structures for training info
        score, steps = 0, 0
        actions_taken = []
        agent_prev_obs = dict(obs)
        agent_prev_action = [2] * args.num_trains
        update_values = [False] * args.num_trains
        action_count = [0] * action_size
        action_dict = dict()

        # Do an episode
        for step in range(args.max_moves - 1):
            # Inference step
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    action = policy.act(obs[agent], eps=args.eps_start)
                    update_values[agent] = True
                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    action = 0
                    update_values[agent] = False
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if args.render and episode % args.checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=True
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                # Only learn from timesteps where something happened
                if update_values[agent] or done['__all__']:
                    learn_timer.start()
                    policy.step(
                        agent_prev_obs[agent], agent_prev_action[agent],
                        all_rewards[agent], obs[agent], done[agent]
                    )
                    learn_timer.end()
                    agent_prev_obs[agent] = obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Update observation and score
                obs[agent] = next_obs[agent]
                score += all_rewards[agent]

            # Break if every agent arrived
            steps = step
            if done['__all__']:
                break

        # Epsilon decay
        args.eps_start = max(args.eps_end, args.eps_decay * args.eps_start)

        # Save final scores
        tasks_finished = sum(
            done[i] for i in train_env.get_agent_handles()
        )
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = (
            score / (args.max_moves * train_env.get_num_agents())
        )
        action_probs = action_count / np.sum(action_count)
        smoothed_normalized_score = (
            smoothed_normalized_score * train_smoothing +
            normalized_score * (1.0 - train_smoothing)
        )
        smoothed_completion = (
            smoothed_completion * train_smoothing +
            completion * (1.0 - train_smoothing)
        )

        # Save model and replay buffer at checkpoint
        if episode % args.checkpoint_interval == 0:
            torch.save(
                policy.qnetwork_local, './checkpoints/' +
                training_id + '-' + str(episode) + '.pth'
            )
            if args.save_replay_buffer:
                policy.save_replay_buffer(
                    './replay_buffers/' + training_id +
                    '-' + str(episode) + '.pkl'
                )
            if args.render:
                env_renderer.close_window()

        # Print final episode info
        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action probabilities: {}'.format(
                episode,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                args.eps_start,
                format_action_probabilities(action_probs)
            ), end=" "
        )

        # Evaluate policy and log results at some interval
        if episode % args.checkpoint_interval == 0 and args.n_val_episodes > 0:
            scores, completions, val_steps = eval_policy(
                args, val_env, policy
            )

            # Save final validation scores
            smoothed_val_normalized_score = (
                smoothed_val_normalized_score * val_smoothing +
                np.mean(scores) * (1.0 - val_smoothing)
            )
            smoothed_val_completion = (
                smoothed_val_completion * val_smoothing +
                np.mean(completions) * (1.0 - val_smoothing)
            )

            # Log validation metrics to tensorboard
            tensorboard_log(
                writer, "validation/scores", scores, episode)
            tensorboard_log(
                writer, "validation/completions", completions, episode
            )
            tensorboard_log(
                writer, "validation/steps", val_steps, episode
            )
            tensorboard_log(
                writer, "validation/smoothed_score",
                smoothed_val_normalized_score, episode
            )
            tensorboard_log(
                writer, "validation/smoothed_completion",
                smoothed_val_completion, episode
            )

        # Log training actions info to tensorboard
        tensorboard_log(
            writer, "actions/distribution", np.array(actions_taken), episode
        )
        tensorboard_log(
            writer, "actions/nothing", action_probs[RailEnvActions.DO_NOTHING], episode
        )
        tensorboard_log(
            writer, "actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode
        )
        tensorboard_log(
            writer, "actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode
        )
        tensorboard_log(
            writer, "actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode
        )
        tensorboard_log(
            writer, "actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode
        )

        # Log training info to tensorboard
        tensorboard_log(writer, "training/steps", steps, episode)
        tensorboard_log(writer, "training/epsilon", args.eps_start, episode)
        tensorboard_log(writer, "training/loss", policy.loss, episode)
        tensorboard_log(writer, "training/score", normalized_score, episode)
        tensorboard_log(writer, "training/completion", completion, episode)
        tensorboard_log(
            writer, "training/smoothed_score", smoothed_normalized_score, episode
        )
        tensorboard_log(
            writer, "training/smoothed_completion", smoothed_completion, episode
        )
        tensorboard_log(
            writer, "training/buffer_size", len(policy.memory), episode
        )

        # Log training time info to tensorboard
        tensorboard_log(writer, "utils.Timer/reset",
                        reset_timer.get(), episode)
        tensorboard_log(writer, "utils.Timer/step", step_timer.get(), episode)
        tensorboard_log(writer, "utils.Timer/learn",
                        learn_timer.get(), episode)
        tensorboard_log(
            writer, "utils.Timer/total", training_timer.get_current(), episode
        )

        # Close tensorboard
        writer.close()


def eval_policy(args, env, policy):
    '''
    Perform a validation round with the given policy
    in the specified environment
    '''
    action_dict = dict()
    scores, completions, steps = [], [], []

    # Do the specified number of episodes
    for episode in range(args.n_val_episodes):
        score = 0.0
        final_step = 0
        if args.fix_random:
            obs, info = env.reset(random_seed=RANDOM_SEED)
        else:
            obs, info = env.reset(
                regenerate_rail=True, regenerate_schedule=True,
                random_seed=utils.get_seed(env)
            )
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        # Do an episode
        for step in range(args.max_moves - 1):
            # Perform a step
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    action = policy.act(obs[agent], eps=0.0)
                action_dict.update({agent: action})
            obs, all_rewards, done, info = env.step(action_dict)

            # Update rewards
            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            # Break if every agent arrived
            final_step = step
            if done['__all__']:
                break

        # Save final scores
        scores.append(score / (args.max_moves * env.get_num_agents()))
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completions.append(tasks_finished / max(1, env.get_num_agents()))
        steps.append(final_step)

    # Print validation results
    print(
        "\t✅ Validation: score {:.3f} done {:.1f}%".format(
            np.mean(scores), np.mean(completions) * 100.0
        )
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
        "--width", action='store', default=16,
        help="environment width", type=int
    )
    parser.add_argument(
        "--height", action='store', default=16,
        help="environment height", type=int
    )
    parser.add_argument(
        "--malfunction", action='store_true', default=False,
        help="enable malfunctions"
    )
    parser.add_argument(
        "--malfunction_rate", action='store', default=1 / 10000,
        help="malfunction rate", type=float
    )
    parser.add_argument(
        "--malfunction_min_duration", action='store', default=20,
        help="malfunction minimum duration", type=int
    )
    parser.add_argument(
        "--malfunction_max_duration", action='store', default=50,
        help="malfunction maximum duration", type=int
    )
    parser.add_argument(
        "--max_moves", action='store', default=500,
        help="maximum number of moves in an episode", type=int
    )
    parser.add_argument(
        "--max_depth", action='store', default=5,
        help="predictor maximum depth", type=int
    )
    parser.add_argument(
        "--max_cities", action='store', default=3,
        help="maximum number of cities where agents can start or end", type=int
    )
    parser.add_argument(
        "--grid", action='store_true',
        help="type of city distribution"
    )
    parser.add_argument(
        "--max_rails_between_cities", action='store', default=4,
        help="maximum number of tracks allowed between cities", type=int
    )
    parser.add_argument(
        "--max_rails_in_cities", action='store', default=4,
        help="maximum number of parallel tracks within a city", type=int
    )
    parser.add_argument(
        "--variable_speed", action='store_true', default=False,
        help="enable variable speed"
    )

    # Training parameters
    parser.add_argument(
        "--n_train_episodes", action='store', default=2500,
        help="number of episodes to run", type=int
    )
    parser.add_argument(
        "--n_val_episodes", action='store', default=25,
        help="number of evaluation episodes", type=int
    )
    parser.add_argument(
        "--checkpoint_interval", action='store', default=100,
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
        "--eps_decay", action='store', default=0.99,
        help="exploration decay", type=float
    )
    parser.add_argument(
        "--buffer_size", action='store', default=int(1e5),
        help="replay buffer size", type=int
    )
    parser.add_argument(
        "--buffer_min_size", action='store', default=0,
        help="minimum buffer size to start training", type=int
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
        "--batch_size", action='store', default=128,
        help="minibatch size", type=int
    )
    parser.add_argument(
        "--gamma", action='store', default=0.99,
        help="discount factor", type=float
    )
    parser.add_argument(
        "--tau", action='store', default=1e-3,
        help="soft update of target parameters", type=float
    )
    parser.add_argument(
        "--learning_rate", action='store', default=0.5e-4,
        help="learning rate", type=float
    )
    parser.add_argument(
        "--hidden_size", default=128,
        help="hidden size (2 fc layers)", type=int
    )
    parser.add_argument(
        "--update_every", action='store', default=8,
        help="how often to update the network", type=int
    )
    parser.add_argument(
        "--use_gpu", action='store_true',
        help="use GPU if available"
    )
    parser.add_argument(
        "--num_threads", action='store', default=1,
        help="number of threads PyTorch can use", type=int
    )
    parser.add_argument(
        "--render", action='store_true',
        help="render 1 episode in 100"
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
