import time
from datetime import datetime

import yaml
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.rail_env import RailEnvActions

import utils
import env_utils
import policies
from action_selectors import ACTION_SELECTORS, PARAMETER_DECAYS
from env_utils import RailEnvChoices
from policies import POLICIES


def tensorboard_log(writer, name, x, y, plot=['min', 'max', 'mean', 'std', 'hist']):
    '''
    Log the given x/y values to tensorboard
    '''
    if not isinstance(x, np.ndarray) and not isinstance(x, list):
        writer.add_scalar(name, x, y)
    else:
        if ((isinstance(x, list) and len(x) == 0) or
                (isinstance(x, np.ndarray) and x.size == 0)):
            return
        if 'min' in plot:
            writer.add_scalar(f"{name}_min", np.min(x), y)
        if 'max' in plot:
            writer.add_scalar(f"{name}_max", np.max(x), y)
        if 'mean' in plot:
            writer.add_scalar(f"{name}_mean", np.mean(x), y)
        if 'std' in plot:
            writer.add_scalar(f"{name}_std", np.std(x), y)
        if 'hist' in plot:
            writer.add_histogram(name, np.array(x), y)


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


def train_agents(args, writer):
    '''
    Train and evaluate agents on the specified environments
    '''
    # Initialize threads and seeds
    utils.set_num_threads(args.generic.num_threads)
    if args.generic.fix_random:
        utils.fix_random(args.generic.random_seed)

    # Setup the environments
    train_env = env_utils.create_rail_env(
        args, load_env=args.training.train_env.load
    )
    eval_env = env_utils.create_rail_env(
        args, load_env=args.training.eval_env.load
    )

    # Define "static" random seeds for evaluation purposes
    eval_seeds = [args.env.seed] * args.training.eval_env.episodes
    if args.training.eval_env.all_random:
        eval_seeds = [
            env_utils.get_seed(eval_env)
            for e in range(args.training.eval_env.episodes)
        ]

    # Pick action selector and parameter decay
    pd_type = args.parameter_decay.type.get_true_key()
    parameter_decay = PARAMETER_DECAYS[pd_type](
        parameter_start=args.parameter_decay.start,
        parameter_end=args.parameter_decay.end,
        total_episodes=args.training.train_env.episodes,
        decaying_episodes=args.parameter_decay.decaying_episodes
    )
    as_type = args.action_selector.type.get_true_key()
    action_selector = ACTION_SELECTORS[as_type](parameter_decay)

    # Initialize the agents policy
    policy_type = args.policy.type.get_true_key()
    policy = POLICIES[policy_type](
        args, train_env.state_size, action_selector, training=True
    )
    if args.generic.wandb_gradients:
        policy.enable_wandb()

    # Handle replay buffer
    if args.replay_buffer.load:
        try:
            policy.load_replay_buffer(args.replay_buffer.load)
        except RuntimeError as e:
            print(
                "\n🛑 Could't load replay buffer, were the experiences generated using the same depth?"
            )
            print(e)
            exit(1)
    print("\n💾 Replay buffer status: {}/{} experiences".format(
        len(policy.memory), args.replay_buffer.size
    ))

    # Set the unique ID for this training
    now = datetime.now()
    training_id = now.strftime('%Y%m%d-%H%M%S')

    # Print initial training info
    training_timer = utils.Timer()
    training_timer.start()
    print("\n🚉 Starting training \t Training {} trains on {}x{} grid for {} episodes \tEvaluating on {} episodes every {} episodes".format(
        args.env.num_trains,
        args.env.width, args.env.height,
        args.training.train_env.episodes,
        args.training.eval_env.episodes,
        args.training.checkpoint
    ))
    print(f"\n🧠 Model with training id {training_id}\n")

    # Do the specified number of episodes
    avg_score, avg_completion = 0.0, 0.0
    for episode in range(args.training.train_env.episodes + 1):
        # Initialize timers
        step_timer = utils.Timer()
        reset_timer = utils.Timer()
        learn_timer = utils.Timer()
        inference_timer = utils.Timer()

        # Reset environment and renderer
        reset_timer.start()
        if not args.training.train_env.all_random:
            obs, info = train_env.reset(random_seed=args.env.seed)
        else:
            obs, info = train_env.reset(
                regenerate_rail=True, regenerate_schedule=True,
                random_seed=env_utils.get_seed(train_env)
            )
        reset_timer.end()
        if args.training.renderer.training and episode % args.training.renderer.train_checkpoint == 0:
            env_renderer = train_env.get_renderer()

        # Compute agents with same source
        agents_with_same_start = train_env.get_agents_same_start()

        # Initialize data structures for training info
        score, steps = 0, 0
        choices_taken = []
        choices_count = [0] * policies.CHOICE_SIZE
        num_exploration_choices = [0] * policies.CHOICE_SIZE
        legal_choices = dict()
        update_values = [False] * args.env.num_trains
        action_dict, choice_dict = dict(), dict()
        rewards = [0] * args.env.num_trains
        arrived_agents = set()
        prev_obs = dict()
        prev_choices = [RailEnvChoices.CHOICE_LEFT.value] * args.env.num_trains
        for handle in range(args.env.num_trains):
            legal_choices[handle] = train_env.railway_encoding.get_legal_choices(
                handle, train_env.railway_encoding.get_agent_actions(handle)
            )
            if obs[handle] is not None:
                prev_obs[handle] = env_utils.copy_obs(obs[handle])

        # Do an episode
        for step in range(args.env.max_moves):

            # Prioritize entry of faster agents in the environment
            for position in agents_with_same_start:
                if len(agents_with_same_start[position]) > 0:
                    del agents_with_same_start[position][0]
                    for agent in agents_with_same_start[position]:
                        info['action_required'][agent] = False

            # Compute an action for each agent, if necessary
            inference_timer.start()
            for agent in train_env.get_agent_handles():
                # An action is not required if the train hasn't joined the railway network,
                # if it already reached its target, or if it's currently malfunctioning,
                # or if it's in deadlock or if it's in the middle of traversing a cell
                update_values[agent] = False
                action = RailEnvActions.DO_NOTHING.value
                if info['action_required'][agent]:
                    if train_env.railway_encoding.is_real_decision(agent):
                        legal_actions = train_env.railway_encoding.get_agent_actions(
                            agent
                        )
                        legal_choices[agent] = train_env.railway_encoding.get_legal_choices(
                            agent, legal_actions
                        )
                        choice, is_best = policy.act(
                            obs[agent], legal_choices[agent], training=True
                        )
                        action = train_env.railway_encoding.map_choice_to_action(
                            choice, legal_actions
                        )
                        assert action != RailEnvActions.DO_NOTHING.value, (
                            choice, legal_actions
                        )
                        update_values[agent] = True
                        choices_count[choice] += 1
                        choices_taken.append(choice)
                        num_exploration_choices[choice] += int(is_best)
                        choice_dict.update({agent: choice})
                    else:
                        actions = train_env.railway_encoding.get_agent_actions(
                            agent
                        )
                        assert len(actions) == 1, actions
                        action = actions[0]
                action_dict.update({agent: action})
            inference_timer.end()

            # Environment step
            step_timer.start()
            next_obs, all_rewards, done, info = train_env.step(action_dict)
            step_timer.end()

            # Render an episode at some interval
            if args.training.renderer.training and episode % args.training.renderer.train_checkpoint == 0:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=True, show_rowcols=True
                )

            # Update replay buffer and train agent
            for agent in train_env.get_agent_handles():

                # Accumulate rewards for choices if the agent is not arrived,
                # otherwise return a positive reward equal to the maximum
                # number of steps
                if done[agent]:
                    rewards[agent] = args.env.max_moves
                else:
                    rewards[agent] += all_rewards[agent]

                # Only learn from timesteps where something happened
                if update_values[agent] or (done[agent] and agent not in arrived_agents):
                    learn_timer.start()
                    experience = (
                        prev_obs[agent], prev_choices[agent], rewards[agent],
                        obs[agent], legal_choices[agent], done[agent]
                    )
                    policy.step(experience)
                    learn_timer.end()
                    rewards[agent] = 0
                    prev_obs[agent] = env_utils.copy_obs(obs[agent])
                    prev_choices[agent] = choice_dict[agent]

                # Add agent to the list of arrived agents
                if done[agent]:
                    arrived_agents.add(agent)

                # Update observation and score
                if next_obs[agent] is not None:
                    obs[agent] = env_utils.copy_obs(next_obs[agent])
                score += all_rewards[agent]

            # Break if every agent arrived
            steps = step
            if done['__all__'] or train_env.check_if_all_blocked():
                break

        # Close window
        if args.training.renderer.training and episode % args.training.renderer.train_checkpoint == 0:
            env_renderer.close_window()

        # Parameter decay
        policy.choice_selector.decay()

        # Save final scores
        tasks_finished = sum(
            done[i] for i in train_env.get_agent_handles()
        )
        completion = tasks_finished / train_env.get_num_agents()
        normalized_score = (
            score / (args.env.max_moves * train_env.get_num_agents())
        )
        avg_completion = (
            episode * avg_completion + completion
        ) / (episode + 1)
        avg_score = (episode * avg_score + normalized_score) / (episode + 1)
        choices_probs = choices_count / np.sum(choices_count)

        # Save model and replay buffer at checkpoint
        if episode % args.training.checkpoint == 0:
            policy.save(f'./checkpoints/{training_id}-{episode}')
            # Save partial model to wandb
            if episode > 0 and episode % args.generic.wandb_checkpoint == 0:
                wandb.save(f'./checkpoints/{training_id}-{episode}.local')
            if args.replay_buffer.save:
                policy.save_replay_buffer(
                    f'./replay_buffers/{training_id}-{episode}.pkl'
                )

        # Print episode info
        print(
            '\r🚂 Episode {:4n}'
            '\t 🏆 Score: {:<+4.3f}'
            ' Avg: {:>+4.3f}'
            '\t 💯 Done: {:<7.2%}'
            ' Avg: {:>7.2%}'
            '\t 🦶 Steps: {:3n}'
            '\t 🎲 Epsilon: {:4.3f} '
            '\t 🤔 Choices: {:3n}'
            '\t 🤠 Exploration: {:3n}'
            '\t 🔀 Choices probabilities: {:^}'.format(
                episode,
                normalized_score,
                avg_score,
                completion,
                avg_completion,
                steps,
                policy.choice_selector.epsilon,
                np.sum(choices_count),
                np.sum(num_exploration_choices),
                format_choices_probabilities(choices_probs)
            ), end="\n"
        )

        # Evaluate policy and log results at some interval
        if episode > 0 and episode % args.training.checkpoint == 0 and args.training.eval_env.episodes > 0:
            eval_policy(args, writer, eval_env, policy, eval_seeds, episode)

        # Log training actions info to tensorboard
        tensorboard_log(
            writer, "choices/left",
            choices_probs[env_utils.RailEnvChoices.CHOICE_LEFT.value], episode
        )
        tensorboard_log(
            writer, "choices/right",
            choices_probs[env_utils.RailEnvChoices.CHOICE_RIGHT.value], episode
        )
        tensorboard_log(
            writer, "choices/stop",
            choices_probs[env_utils.RailEnvChoices.STOP.value], episode
        )

        # Log training info to tensorboard
        tensorboard_log(writer, "training/steps", steps, episode)
        tensorboard_log(
            writer, "training/choices_count",
            np.sum(choices_count), episode
        )
        tensorboard_log(
            writer, "training/exploration_choices",
            np.sum(num_exploration_choices), episode
        )
        tensorboard_log(
            writer, "training/epsilon",
            policy.choice_selector.epsilon, episode
        )
        tensorboard_log(
            writer, "training/loss",
            policy.loss.data.item(), episode
        )
        tensorboard_log(writer, "training/score", normalized_score, episode)
        tensorboard_log(writer, "training/completion", completion, episode)
        tensorboard_log(
            writer, "training/buffer_size",
            len(policy.memory), episode
        )

        # Log training time info to tensorboard
        tensorboard_log(writer, "timer/reset", reset_timer.get(), episode)
        tensorboard_log(writer, "timer/step", step_timer.get(), episode)
        tensorboard_log(writer, "timer/learn", learn_timer.get(), episode)
        tensorboard_log(
            writer, "timer/total",
            training_timer.get_current(), episode
        )

    # Print final training info
    print("\n\r🏁 Training Ended \tTrained {} trains on {}x{} grid for {} episodes \t Evaluated on {} episodes every {} episodes".format(
        args.env.num_trains,
        args.env.width, args.env.height,
        args.training.train_env.episodes,
        args.training.eval_env.episodes,
        args.training.checkpoint
    ))
    print(
        f"\n💾 Replay buffer status: {len(policy.memory)}/{args.replay_buffer.size} experiences"
    )

    # Save trained models
    print(f"\n🧠 Saving model with training id {training_id}")
    policy.save(f'./checkpoints/{training_id}-latest')
    wandb.save(f'./checkpoints/{training_id}-latest.local')
    if args.replay_buffer.save:
        policy.save_replay_buffer(
            f'./replay_buffers/{training_id}-latest.pkl'
        )


def eval_policy(args, writer, env, policy, eval_seeds, train_episode):
    '''
    Perform a validation round with the given policy
    in the specified environment
    '''
    action_dict = dict()
    scores, completions, steps, choices_count = [], [], [], []

    # Do the specified number of episodes
    print('\nStarting validation:')
    for episode, seed in enumerate(eval_seeds):
        score = 0.0
        final_step = 0
        choices_taken = []

        # Reset environment and renderer
        if not args.training.eval_env.all_random:
            obs, info = env.reset(random_seed=seed)
        else:
            obs, info = env.reset(
                regenerate_rail=True, regenerate_schedule=True,
                random_seed=seed
            )
        if args.training.renderer.evaluation and episode % args.training.renderer.eval_checkpoint == 0:
            env_renderer = env.get_renderer()

        # Compute agents with same source
        agents_with_same_start = env.get_agents_same_start()

        # Do an episode
        for step in range(args.env.max_moves):

            # Prioritize enter of faster agent in the environment
            for position in agents_with_same_start:
                if len(agents_with_same_start[position]) > 0:
                    del agents_with_same_start[position][0]
                    for agent in agents_with_same_start[position]:
                        info['action_required'][agent] = False

            # Perform a step
            for agent in env.get_agent_handles():
                action = RailEnvActions.DO_NOTHING.value
                if info['action_required'][agent]:
                    if env.railway_encoding.is_real_decision(agent):
                        legal_actions = env.railway_encoding.get_agent_actions(
                            agent
                        )
                        legal_choices = env.railway_encoding.get_legal_choices(
                            agent, legal_actions
                        )
                        choice, is_best = policy.act(
                            obs[agent], legal_choices[agent], training=False
                        )
                        choices_taken.append(choice)
                        action = env.railway_encoding.map_choice_to_action(
                            choice, legal_actions
                        )
                        assert action != RailEnvActions.DO_NOTHING.value, (
                            choice, legal_actions
                        )
                    else:
                        actions = env.railway_encoding.get_agent_actions(agent)
                        assert len(actions) == 1, actions
                        action = actions[0]
                action_dict.update({agent: action})

            # Perform a step in the environment
            obs, all_rewards, done, info = env.step(action_dict)

            # Render an episode at some interval
            if args.training.renderer.evaluation and episode % args.training.renderer.eval_checkpoint == 0:
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
        if args.training.renderer.evaluation and episode % args.training.renderer.eval_checkpoint == 0:
            env_renderer.close_window()

        # Save final scores
        normalized_score = (
            score / (args.env.max_moves * env.get_num_agents())
        )
        scores.append(normalized_score)
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / env.get_num_agents()
        completions.append(completion)
        steps.append(final_step)
        choices_count.append(len(choices_taken))

        # Print evaluation results on one episode
        print(
            '\r🚂 Validation {:3n}'
            '\t 🏆 Score: {:+4.3f}'
            '\t 💯 Done: {:7.2%}'
            '\t 🦶 Steps: {:4n}'
            '\t 🤔 Choices taken {:3n}'.format(
                episode,
                normalized_score,
                completion,
                final_step,
                len(choices_taken)
            ), end="\n"
        )

    # Print validation results
    print(
        '\r✅ Validation End'
        '\t 🏆 Avg score: {:+1.3f}'
        '\t 💯 Avg done: {:7.1%}%'
        '\t 🦶 Avg steps: {:5.2f}'
        '\t 🤔 Avg choices taken {:5.2f}'.format(
            np.mean(scores),
            np.mean(completions),
            np.mean(steps),
            np.mean(choices_count)
        ), end="\n\n"
    )

    # Log validation metrics to tensorboard
    tensorboard_log(
        writer, "evaluation/scores", scores, train_episode,
        plot=['mean', 'std', 'hist']
    )
    tensorboard_log(
        writer, "evaluation/completions", completions, train_episode,
        plot=['mean', 'std', 'hist']
    )
    tensorboard_log(
        writer, "evaluation/steps", steps, train_episode,
        plot=['mean', 'std', 'hist']
    )
    tensorboard_log(
        writer, "evaluation/choices_count", choices_count, train_episode,
        plot=['mean', 'std', 'hist']
    )


def main():
    '''
    Train environment with custom observation and prediction
    '''
    with open('parameters.yml', 'r') as conf:
        args = yaml.load(conf, Loader=yaml.FullLoader)
    wandb.init(
        project='flatland-challenge',
        entity="wadaboa",
        config=args
    )
    wandb.tensorboard.patch(tensorboardX=False, pytorch=True)
    writer = SummaryWriter()
    args = utils.Struct(**args)
    train_agents(args, writer)
    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
