from models import DDDQNPolicy
from utils import Timer
from datetime import datetime
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import psutil
from flatland.utils.rendertools import RenderTool
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from predictions import ShortestPathPredictor
from observations import CustomObservation

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


RANDOM_SEED = 42


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
            seed=RANDOM_SEED,
            grid_mode=args.grid,
            max_rails_between_cities=args.max_rails_between_cities,
            max_rails_in_city=args.max_rails_in_cities,
        )

    # Initialize predictor and observer
    predictor = ShortestPathPredictor(max_depth=args.max_depth)
    observation_builder = CustomObservation(
        max_depth=args.max_depth, predictor=predictor
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
        remove_agents_at_target=True
    )


def set_num_threads(num_threads):
    '''
    Set the maximum number of threads PyTorch can use
    '''
    torch.set_num_threads(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(training_params.num_threads)


def fix_random(seed):
    '''
    Fix all the possible sources of randomness
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_agents(args):
    '''
    Train and evaluate agents on the specified environments
    '''
    set_num_threads(args.num_threads)
    if args.fix_random:
        fix_random(RANDOM_SEED)

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Setup the environments
    train_env = create_rail_env(args, env=args.train_env)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    val_env = create_rail_env(args, env=args.val_env)
    val_env.reset(regenerate_schedule=True, regenerate_rail=True)

    # Setup renderer
    if args.render:
        env_renderer = RenderTool(
            train_env,
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            screen_height=600,
            screen_width=800
        )

    # Compute state size and action size
    state_size = (args.max_depth ** 2) * observation_builder.FEATURES
    action_size = 5

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * args.num_trains
    agent_prev_obs = [None] * args.num_trains
    agent_prev_action = [2] * args.num_trains
    update_values = [False] * args.num_trains

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, action_size, train_params)

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
        len(policy.memory.memory), train_params.buffer_size)
    )

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(train_env_params), {})
    writer.add_hparams(vars(obs_params), {})

    training_timer = Timer()
    training_timer.start()
    print("\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
        train_env.get_num_agents(),
        args.width, args.height,
        args.n_train_episodes,
        args.n_val_episodes,
        args.checkpoint_interval,
        training_id
    ))

    for episode_idx in range(args.n_train_episodes + 1):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = train_env.reset(
            regenerate_rail=True, regenerate_schedule=True
        )
        reset_timer.end()

        if args.render:
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []
        agent_prev_obs = list(obs)

        # Run episode
        for step in range(args.max_moves - 1):
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=args.eps_start)

                    action_count[action] += 1
                    actions_taken.append(action)
                else:
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if is currently malfunctioning.
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if args.render and episode_idx % args.checkpoint_interval == 0:
                env_renderer.render_env(
                    show=True,
                    frames=False,
                    show_observations=False,
                    show_predictions=False
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent],
                                all_rewards[agent], agent_obs[agent], done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent] is not None:
                    preproc_timer.start()
                    # agent_obs[agent] = normalize_observation(
                    #    next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                    agent_obs[agent] = next_obs[agent]
                    preproc_timer.end()

                score += all_rewards[agent]

            nb_steps = step

            if done['__all__']:
                break

        # Epsilon decay
        args.eps_start = max(args.eps_end, args.eps_decay * args.eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx]
                             for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / \
            (args.max_moves * train_env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * \
            smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * \
            smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode_idx % args.checkpoint_interval == 0:
            torch.save(policy.qnetwork_local, './checkpoints/' +
                       training_id + '-' + str(episode_idx) + '.pth')

            if args.save_replay_buffer:
                policy.save_replay_buffer(
                    './replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            if args.render:
                env_renderer.close_window()

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                args.eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if episode_idx % args.checkpoint_interval == 0 and args.n_val_episodes > 0:
            scores, completions, nb_steps_eval = eval_policy(
                eval_env, policy, train_params, obs_params)

            writer.add_scalar("evaluation/scores_min",
                              np.min(scores), episode_idx)
            writer.add_scalar("evaluation/scores_max",
                              np.max(scores), episode_idx)
            writer.add_scalar("evaluation/scores_mean",
                              np.mean(scores), episode_idx)
            writer.add_scalar("evaluation/scores_std",
                              np.std(scores), episode_idx)
            writer.add_histogram("evaluation/scores",
                                 np.array(scores), episode_idx)
            writer.add_scalar("evaluation/completions_min",
                              np.min(completions), episode_idx)
            writer.add_scalar("evaluation/completions_max",
                              np.max(completions), episode_idx)
            writer.add_scalar("evaluation/completions_mean",
                              np.mean(completions), episode_idx)
            writer.add_scalar("evaluation/completions_std",
                              np.std(completions), episode_idx)
            writer.add_histogram("evaluation/completions",
                                 np.array(completions), episode_idx)
            writer.add_scalar("evaluation/nb_steps_min",
                              np.min(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_max",
                              np.max(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_mean",
                              np.mean(nb_steps_eval), episode_idx)
            writer.add_scalar("evaluation/nb_steps_std",
                              np.std(nb_steps_eval), episode_idx)
            writer.add_histogram("evaluation/nb_steps",
                                 np.array(nb_steps_eval), episode_idx)

            smoothing = 0.9
            smoothed_eval_normalized_score = smoothed_eval_normalized_score * \
                smoothing + np.mean(scores) * (1.0 - smoothing)
            smoothed_eval_completion = smoothed_eval_completion * \
                smoothing + np.mean(completions) * (1.0 - smoothing)
            writer.add_scalar("evaluation/smoothed_score",
                              smoothed_eval_normalized_score, episode_idx)
            writer.add_scalar("evaluation/smoothed_completion",
                              smoothed_eval_completion, episode_idx)

        # Save logs to tensorboard
        writer.add_scalar("training/score", normalized_score, episode_idx)
        writer.add_scalar("training/smoothed_score",
                          smoothed_normalized_score, episode_idx)
        writer.add_scalar("training/completion",
                          np.mean(completion), episode_idx)
        writer.add_scalar("training/smoothed_completion",
                          np.mean(smoothed_completion), episode_idx)
        writer.add_scalar("training/nb_steps", nb_steps, episode_idx)
        writer.add_histogram("actions/distribution",
                             np.array(actions_taken), episode_idx)
        writer.add_scalar("actions/nothing",
                          action_probs[RailEnvActions.DO_NOTHING], episode_idx)
        writer.add_scalar(
            "actions/left", action_probs[RailEnvActions.MOVE_LEFT], episode_idx)
        writer.add_scalar(
            "actions/forward", action_probs[RailEnvActions.MOVE_FORWARD], episode_idx)
        writer.add_scalar(
            "actions/right", action_probs[RailEnvActions.MOVE_RIGHT], episode_idx)
        writer.add_scalar(
            "actions/stop", action_probs[RailEnvActions.STOP_MOVING], episode_idx)
        writer.add_scalar("training/epsilon", args.eps_start, episode_idx)
        writer.add_scalar("training/buffer_size",
                          len(policy.memory), episode_idx)
        writer.add_scalar("training/loss", policy.loss, episode_idx)
        writer.add_scalar("timer/reset", reset_timer.get(), episode_idx)
        writer.add_scalar("timer/step", step_timer.get(), episode_idx)
        writer.add_scalar("timer/learn", learn_timer.get(), episode_idx)
        writer.add_scalar("timer/preproc", preproc_timer.get(), episode_idx)
        writer.add_scalar(
            "timer/total", training_timer.get_current(), episode_idx)


def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def eval_policy(args, env, policy):
    action_dict = dict()
    scores = []
    completions = []
    steps = []

    # Do the specified number of episodes
    for episode_idx in range(args.n_val_episodes):
        score = 0.0
        final_step = 0
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
        normalized_score = score / (args.max_moves * env.get_num_agents())
        scores.append(normalized_score)
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)
        steps.append(final_step)

    # Print validation results
    print(
        "\t✅ Eval: score {:.3f} done {:.1f}%".format(
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
        "--buffer_size", action='store', default=int(1e5)
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
        "--save_replay_buffer", action='store_true'
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
        "--hidden_size", default=128
        help="hidden size (2 fc layers)", type=int
    )
    parser.add_argument(
        "--update_every", action='store', default=8,
        help="how often to update the network", type=int
    )
    parser.add_argument(
        "--use_gpu", action='store_true'
        help="use GPU if available"
    )
    parser.add_argument(
        "--num_threads", action='store', default=1,
        help="number of threads PyTorch can use", type=int
    )
    parser.add_argument(
        "--render", action='store_true'
        help="render 1 episode in 100"
    )
    parser.add_argument(
        "--fix_random", action='store_true'
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
