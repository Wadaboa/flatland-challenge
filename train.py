from models import DDDQNPolicy
from observation_utils import normalize_observation
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
from flatland.envs.observations import TreeObsForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))


try:
    import wandb

    wandb.init(sync_tensorboard=True)
except ImportError:
    print("Install wandb to log to Weights & Biases")


def create_rail_env(env_params, tree_observation):
    n_agents = env_params.n_agents
    x_dim = env_params.x_dim
    y_dim = env_params.y_dim
    n_cities = env_params.n_cities
    max_rails_between_cities = env_params.max_rails_between_cities
    max_rails_in_city = env_params.max_rails_in_city
    seed = env_params.seed

    # Break agents from time to time
    malfunction_parameters = {'malfunction_rate': env_params.malfunction_rate,  # Rate of malfunction occurence of single agent
                              'min_duration': 20,  # Minimal duration of malfunction
                              'max_duration': 50  # Max duration of malfunction
                              }

    return RailEnv(
        width=x_dim, height=y_dim,
        rail_generator=sparse_rail_generator(
            max_num_cities=n_cities,
            grid_mode=False,
            max_rails_between_cities=max_rails_between_cities,
            max_rails_in_city=max_rails_in_city
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        malfunction_generator_and_process_data=malfunction_from_params(
            malfunction_parameters),
        obs_builder_object=tree_observation,
        random_seed=seed
    )


def train_agent(train_params, train_env_params, eval_env_params, obs_params):
    # Environment parameters
    n_agents = train_env_params.n_agents
    x_dim = train_env_params.x_dim
    y_dim = train_env_params.y_dim
    n_cities = train_env_params.n_cities
    max_rails_between_cities = train_env_params.max_rails_between_cities
    max_rails_in_city = train_env_params.max_rails_in_city
    seed = train_env_params.seed

    # Unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    # Observation parameters
    observation_tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius
    observation_max_path_depth = obs_params.observation_max_path_depth

    # Training parameters
    eps_start = train_params.eps_start
    eps_end = train_params.eps_end
    eps_decay = train_params.eps_decay
    n_episodes = train_params.n_episodes
    checkpoint_interval = train_params.checkpoint_interval
    n_eval_episodes = train_params.n_evaluation_episodes
    restore_replay_buffer = train_params.restore_replay_buffer
    save_replay_buffer = train_params.save_replay_buffer

    # Set the seeds
    random.seed(seed)
    np.random.seed(seed)

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    tree_observation = TreeObsForRailEnv(
        max_depth=observation_tree_depth, predictor=predictor)

    # Setup the environments
    train_env = create_rail_env(train_env_params, tree_observation)
    train_env.reset(regenerate_schedule=True, regenerate_rail=True)
    eval_env = create_rail_env(eval_env_params, tree_observation)
    eval_env.reset(regenerate_schedule=True, regenerate_rail=True)

    # Setup renderer
    if train_params.render:
        env_renderer = RenderTool(train_env, gl="PGL")

    # Calculate the state size given the depth of the tree observation and the number of features
    n_features_per_node = train_env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # Max number of steps per episode
    # This is the official formula used during evaluations
    # See details in flatland.envs.schedule_generators.sparse_schedule_generator
    # max_steps = int(4 * 2 * (env.height + env.width + (n_agents / n_cities)))
    max_steps = train_env._max_episode_steps

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [2] * n_agents
    update_values = [False] * n_agents

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    # Double Dueling DQN policy
    policy = DDDQNPolicy(state_size, action_size, train_params)

    # Loads existing replay buffer
    if restore_replay_buffer:
        try:
            policy.load_replay_buffer(restore_replay_buffer)
            policy.test()
        except RuntimeError as e:
            print(
                "\n🛑 Could't load replay buffer, were the experiences generated using the same tree depth?")
            print(e)
            exit(1)

    print("\n💾 Replay buffer status: {}/{} experiences".format(
        len(policy.memory.memory), train_params.buffer_size))

    hdd = psutil.disk_usage('/')
    if save_replay_buffer and (hdd.free / (2 ** 30)) < 500.0:
        print("⚠️  Careful! Saving replay buffers will quickly consume a lot of disk space. You have {:.2f}gb left.".format(
            hdd.free / (2 ** 30)))

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(train_env_params), {})
    writer.add_hparams(vars(obs_params), {})

    training_timer = Timer()
    training_timer.start()

    print("\n🚉 Training {} trains on {}x{} grid for {} episodes, evaluating on {} episodes every {} episodes. Training id '{}'.\n".format(
        train_env.get_num_agents(),
        x_dim, y_dim,
        n_episodes,
        n_eval_episodes,
        checkpoint_interval,
        training_id
    ))

    for episode_idx in range(n_episodes + 1):
        step_timer = Timer()
        reset_timer = Timer()
        learn_timer = Timer()
        preproc_timer = Timer()
        inference_timer = Timer()

        # Reset environment
        reset_timer.start()
        obs, info = train_env.reset(
            regenerate_rail=True, regenerate_schedule=True)
        reset_timer.end()

        if train_params.render:
            env_renderer.set_new_rail()

        score = 0
        nb_steps = 0
        actions_taken = []

        # Build initial agent-specific observations
        for agent in train_env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(
                    obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()

        # Run episode
        for step in range(max_steps - 1):
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=eps_start)

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
            if train_params.render and episode_idx % checkpoint_interval == 0:
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
                if next_obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(
                        next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]

            nb_steps = step

            if done['__all__']:
                break

        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx]
                             for idx in train_env.get_agent_handles())
        completion = tasks_finished / max(1, train_env.get_num_agents())
        normalized_score = score / (max_steps * train_env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * \
            smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * \
            smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            torch.save(policy.qnetwork_local, './checkpoints/' +
                       training_id + '-' + str(episode_idx) + '.pth')

            if save_replay_buffer:
                policy.save_replay_buffer(
                    './replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')

            if train_params.render:
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
                eps_start,
                format_action_prob(action_probs)
            ), end=" ")

        # Evaluate policy and log results at some interval
        if episode_idx % checkpoint_interval == 0 and n_eval_episodes > 0:
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
        writer.add_scalar("training/epsilon", eps_start, episode_idx)
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


def eval_policy(env, policy, train_params, obs_params):
    n_eval_episodes = train_params.n_evaluation_episodes
    max_steps = env._max_episode_steps
    tree_depth = obs_params.observation_tree_depth
    observation_radius = obs_params.observation_radius

    action_dict = dict()
    scores = []
    completions = []
    nb_steps = []

    for episode_idx in range(n_eval_episodes):
        agent_obs = [None] * env.get_num_agents()
        score = 0.0

        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        final_step = 0

        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = normalize_observation(
                        obs[agent], tree_depth=tree_depth, observation_radius=observation_radius)

                action = 0
                if info['action_required'][agent]:
                    action = policy.act(agent_obs[agent], eps=0.0)
                action_dict.update({agent: action})

            obs, all_rewards, done, info = env.step(action_dict)

            for agent in env.get_agent_handles():
                score += all_rewards[agent]

            final_step = step

            if done['__all__']:
                break

        normalized_score = score / (max_steps * env.get_num_agents())
        scores.append(normalized_score)

        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        completions.append(completion)

        nb_steps.append(final_step)

    print("\t✅ Eval: score {:.3f} done {:.1f}%".format(
        np.mean(scores), np.mean(completions) * 100.0))

    return scores, completions, nb_steps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--n_episodes", help="number of episodes to run", default=2500, type=int)
    parser.add_argument("-t", "--training_env_config",
                        help="training config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("-e", "--evaluation_env_config",
                        help="evaluation config id (eg 0 for Test_0)", default=0, type=int)
    parser.add_argument("--n_evaluation_episodes",
                        help="number of evaluation episodes", default=25, type=int)
    parser.add_argument("--checkpoint_interval",
                        help="checkpoint interval", default=100, type=int)
    parser.add_argument("--eps_start", help="max exploration",
                        default=1.0, type=float)
    parser.add_argument("--eps_end", help="min exploration",
                        default=0.01, type=float)
    parser.add_argument(
        "--eps_decay", help="exploration decay", default=0.99, type=float)
    parser.add_argument(
        "--buffer_size", help="replay buffer size", default=int(1e5), type=int)
    parser.add_argument(
        "--buffer_min_size", help="min buffer size to start training", default=0, type=int)
    parser.add_argument("--restore_replay_buffer",
                        help="replay buffer to restore", default="", type=str)
    parser.add_argument("--save_replay_buffer",
                        help="save replay buffer at each evaluation interval", default=False, type=bool)
    parser.add_argument(
        "--batch_size", help="minibatch size", default=128, type=int)
    parser.add_argument("--gamma", help="discount factor",
                        default=0.99, type=float)
    parser.add_argument(
        "--tau", help="soft update of target parameters", default=1e-3, type=float)
    parser.add_argument("--learning_rate",
                        help="learning rate", default=0.5e-4, type=float)
    parser.add_argument(
        "--hidden_size", help="hidden size (2 fc layers)", default=128, type=int)
    parser.add_argument(
        "--update_every", help="how often to update the network", default=8, type=int)
    parser.add_argument(
        "--use_gpu", help="use GPU if available", default=False, type=bool)
    parser.add_argument(
        "--num_threads", help="number of threads PyTorch can use", default=1, type=int)
    parser.add_argument(
        "--render", help="render 1 episode in 100", default=False, type=bool)
    training_params = parser.parse_args()

    env_params = [
        {
            # Test_0
            "n_agents": 5,
            "x_dim": 25,
            "y_dim": 25,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 50,
            "seed": 0
        },
        {
            # Test_1
            "n_agents": 10,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 2,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 100,
            "seed": 0
        },
        {
            # Test_2
            "n_agents": 20,
            "x_dim": 30,
            "y_dim": 30,
            "n_cities": 3,
            "max_rails_between_cities": 2,
            "max_rails_in_city": 3,
            "malfunction_rate": 1 / 200,
            "seed": 0
        },
    ]

    obs_params = {
        "observation_tree_depth": 2,
        "observation_radius": 10,
        "observation_max_path_depth": 30
    }

    def check_env_config(id):
        if id >= len(env_params) or id < 0:
            print("\n🛑 Invalid environment configuration, only Test_0 to Test_{} are supported.".format(
                len(env_params) - 1))
            exit(1)

    check_env_config(training_params.training_env_config)
    check_env_config(training_params.evaluation_env_config)

    training_env_params = env_params[training_params.training_env_config]
    evaluation_env_params = env_params[training_params.evaluation_env_config]

    print("\nTraining parameters:")
    pprint(vars(training_params))
    print("\nTraining environment parameters (Test_{}):".format(
        training_params.training_env_config))
    pprint(training_env_params)
    print("\nEvaluation environment parameters (Test_{}):".format(
        training_params.evaluation_env_config))
    pprint(evaluation_env_params)
    print("\nObservation parameters:")
    pprint(obs_params)

    os.environ["OMP_NUM_THREADS"] = str(training_params.num_threads)
    train_agent(training_params, Namespace(**training_env_params),
                Namespace(**evaluation_env_params), Namespace(**obs_params))
