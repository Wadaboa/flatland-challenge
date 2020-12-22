import time
import os
from datetime import datetime
from networkx.readwrite.json_graph import adjacency

import yaml
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from flatland.envs.rail_env import RailEnvActions

import utils
from env import env_utils
from policy.action_selectors import ACTION_SELECTORS, PARAMETER_DECAYS
from env.env_utils import RailEnvChoices
from policy.policies import POLICIES


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
    if args.generic.enable_wandb and args.generic.wandb_gradients.enabled:
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
    if args.training.renderer.training and args.training.renderer.save_frames:
        frames_dir = f"tmp/frames/{training_id}"
        os.makedirs(frames_dir, exist_ok=True)

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
    scores, custom_scores, completions, steps, deadlocks = [], [], [], [], []
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
                regenerate_rail=True, regenerate_schedule=True
            )
        reset_timer.end()
        if args.training.renderer.training and episode % args.training.renderer.train_checkpoint == 0:
            env_renderer = train_env.get_renderer()

        # Compute agents with same source
        agents_with_same_start = train_env.get_agents_same_start()

        # Initialize data structures for training info
        score, custom_score, final_step = 0.0, 0.0, 0
        choices_taken = []
        choices_count = [0] * env_utils.RailEnvChoices.choice_size()
        num_exploration_choices = [0] * env_utils.RailEnvChoices.choice_size()
        legal_choices, legal_actions = dict(), dict()
        update_values = [False] * args.env.num_trains
        action_dict, choice_dict = dict(), dict()
        prev_obs = dict()
        done = dict()
        prev_choices = [RailEnvChoices.CHOICE_LEFT.value] * args.env.num_trains
        for handle in range(args.env.num_trains):
            legal_choices[handle] = train_env.railway_encoding.get_legal_choices(
                handle, train_env.railway_encoding.get_agent_actions(handle)
            )
            choice_dict.update({handle: RailEnvChoices.CHOICE_LEFT.value})
            done[handle] = False
            if obs[handle] is not None:
                prev_obs[handle] = env_utils.copy_obs(obs[handle])

        # Do an episode
        for step in range(train_env._max_episode_steps):

            # Prioritize entry of faster agents in the environment
            for position in agents_with_same_start:
                if len(agents_with_same_start[position]) > 0:
                    del agents_with_same_start[position][0]
                    for agent in agents_with_same_start[position]:
                        info['action_required'][agent] = False

            # Compute an action for each agent, if necessary
            inference_timer.start()

            legal_choices, update_values = train_env.pre_act()
            choices, is_best = policy.act(
                list(obs.values()), legal_choices,
                update_values, training=True
            )

            # If the multi agent observation was selected, then the policy.act method
            # should be called with the entire set of observations
            if args.policy.type.multi_agent_graph:
                for agent in train_env.get_agent_handles():
                    legal_choices[agent] = env_utils.RailEnvChoices.default_choices()
                    update_values[agent] = False
                    if info['action_required'][agent]:
                        if train_env.railway_encoding.is_real_decision(agent):
                            legal_actions[agent] = train_env.railway_encoding.get_agent_actions(
                                agent
                            )
                            legal_choices[agent] = train_env.railway_encoding.get_legal_choices(
                                agent, legal_actions[agent]
                            )
                            update_values[agent] = True
                choices, is_best = policy.act(
                    list(obs.values()),
                    np.array(list(legal_choices.values())),
                    ~np.array(update_values), training=True
                )
                for agent in train_env.get_agent_handles():
                    action = RailEnvActions.DO_NOTHING.value
                    if info['action_required'][agent]:
                        if train_env.railway_encoding.is_real_decision(agent):
                            action = train_env.railway_encoding.map_choice_to_action(
                                choices[agent], legal_actions[agent]
                            )
                            assert action != RailEnvActions.DO_NOTHING.value, (
                                choices[agent], legal_actions[agent]
                            )
                            choices_count[choices[agent]] += 1
                            choices_taken.append(choices[agent])
                            num_exploration_choices[choices[agent]] += int(
                                not(is_best[agent])
                            )
                            choice_dict.update({agent: choices[agent]})
                        else:
                            actions = train_env.railway_encoding.get_agent_actions(
                                agent
                            )
                            assert len(actions) == 1, actions
                            action = actions[0]
                    action_dict.update({agent: action})
            # Otherwise, do one call of policy.act for each observation
            else:
                for agent in train_env.get_agent_handles():
                    # An action is not required if the train hasn't joined the railway network,
                    # if it already reached its target, or if it's currently malfunctioning,
                    # or if it's in deadlock or if it's in the middle of traversing a cell
                    update_values[agent] = False
                    legal_choices[agent] = env_utils.RailEnvChoices.default_choices()
                    action = RailEnvActions.DO_NOTHING.value
                    if info['action_required'][agent]:
                        if train_env.railway_encoding.is_real_decision(agent):
                            legal_actions[agent] = train_env.railway_encoding.get_agent_actions(
                                agent
                            )
                            legal_choices[agent] = train_env.railway_encoding.get_legal_choices(
                                agent, legal_actions[agent]
                            )
                            choice, is_best = policy.act(
                                obs[agent], legal_choices[agent], training=True
                            )
                            action = train_env.railway_encoding.map_choice_to_action(
                                choice, legal_actions[agent]
                            )
                            assert action != RailEnvActions.DO_NOTHING.value, (
                                choice, legal_actions[agent]
                            )
                            update_values[agent] = True
                            choices_count[choice] += 1
                            choices_taken.append(choice)
                            num_exploration_choices[choice] += int(
                                not(is_best)
                            )
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
            next_obs, rewards, custom_rewards, done, info = train_env.step(
                action_dict
            )
            step_timer.end()

            # Render an episode at some interval
            if args.training.renderer.training and episode % args.training.renderer.train_checkpoint == 0:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=True, show_rowcols=True
                )
                # Save renderer frame
                if args.training.renderer.save_frames:
                    env_renderer.gl.save_image(
                        "{:s}/{:04d}.png".format(frames_dir, step)
                    )

            # If the multi agent observation was selected, then the policy.step method
            # should be called just one time
            if args.policy.type.multi_agent_graph:
                first_time_finished = [
                    done[agent] and step == train_env.arrived_turns[agent]
                    for agent in train_env.get_agent_handles()
                ]
                first_time_deadlock = [
                    info["deadlocks"][agent] and step == info["deadlock_turns"][agent]
                    for agent in train_env.get_agent_handles()
                ]
                learn_timer.start()
                if any(update_values) or any(first_time_finished) or any(first_time_deadlock):
                    finished = [
                        done[agent] or info["deadlocks"][agent]
                        for agent in train_env.get_agent_handles()
                    ]
                    next_adjacency = train_env.agents_adjacency_matrix(
                        radius=args.observator.max_depth
                    )
                    experience = (
                        np.array(list(prev_obs.values())),
                        np.array(prev_choices),
                        adjacency,
                        np.array(list(custom_rewards.values())),
                        np.array(list(obs.values())),
                        np.array(list(legal_choices.values())),
                        next_adjacency,
                        np.array(finished),
                        ~np.array(update_values)
                    )
                    policy.step(experience)
                learn_timer.end()

                for agent in train_env.get_agent_handles():
                    if update_values[agent] or first_time_finished[agent] or first_time_deadlock[agent]:
                        prev_obs[agent] = env_utils.copy_obs(obs[agent])
                        prev_choices[agent] = choice_dict[agent]

                    # Update observation and score
                    score += rewards[agent]
                    custom_score += custom_rewards[agent]
                    if next_obs[agent] is not None:
                        obs[agent] = env_utils.copy_obs(next_obs[agent])
            # Otherwise, do one call of policy.step for each agent
            else:
                # Update replay buffer and train agent
                for agent in train_env.get_agent_handles():

                    # Only learn from timesteps where something happened
                    if (update_values[agent] or
                            (done[agent] and step == train_env.arrived_turns[agent]) or
                            (info["deadlocks"][agent] and step == info["deadlock_turns"][agent])):
                        learn_timer.start()
                        experience = (
                            prev_obs[agent], prev_choices[agent], custom_rewards[agent],
                            obs[agent], legal_choices[agent],
                            (done[agent] or info["deadlocks"][agent])
                        )
                        policy.step(experience)
                        learn_timer.end()
                        prev_obs[agent] = env_utils.copy_obs(obs[agent])
                        prev_choices[agent] = choice_dict[agent]

                    # Update observation and score
                    score += rewards[agent]
                    custom_score += custom_rewards[agent]
                    if next_obs[agent] is not None:
                        obs[agent] = env_utils.copy_obs(next_obs[agent])

            # Break if every agent arrived
            final_step = step
            if done['__all__'] or train_env.check_if_all_blocked(info["deadlocks"]):
                break

        # Close window
        if args.training.renderer.training and episode % args.training.renderer.train_checkpoint == 0:
            env_renderer.close_window()

        # Parameter decay
        policy.choice_selector.decay()

        # Save final scores
        scores.append(score / (train_env._max_episode_steps *
                               train_env.get_num_agents()))
        custom_scores.append(custom_score / train_env.get_num_agents())
        completions.append(
            sum(done[idx] for idx in train_env.get_agent_handles()) / train_env.get_num_agents())
        steps.append(final_step)
        choices_count.append(len(choices_taken))
        deadlocks.append(
            sum(int(v) for v in info["deadlocks"].values()) / train_env.get_num_agents())
        choices_probs = choices_count / np.sum(choices_count)

        # Save model and replay buffer at checkpoint
        if episode % args.training.checkpoint == 0:
            policy.save(f'./checkpoints/{training_id}-{episode}')
            # Save partial model to wandb
            if args.generic.enable_wandb and episode > 0 and episode % args.generic.wandb_checkpoint == 0:
                wandb.save(f'./checkpoints/{training_id}-{episode}.local')
            if args.replay_buffer.save:
                policy.save_replay_buffer(
                    f'./replay_buffers/{training_id}-{episode}.pkl'
                )

        # Print episode info
        print(
            '\r🚂 Episode {:4n}'
            '\t 🏆 Score: {:<+5.4f}'
            ' Avg: {:>+5.4f}'
            '\t 🏅 Custom score: {:<+5.4f}'
            ' Avg: {:>+5.4f}'
            '\t 💯 Done: {:<7.2%}'
            ' Avg: {:>7.2%}'
            '\t 💀 Deadlocks: {:<7.2%}'
            ' Avg: {:>7.2%}'
            '\t 🦶 Steps: {:4n}/{:4n}'
            '\t 🎲 Exploration prob: {:4.3f} '
            '\t 🤔 Choices: {:4n}'
            '\t 🤠 Exploration: {:3n}'
            '\t 🔀 Choices probs: {:^}'.format(
                episode,
                scores[-1],
                np.mean(scores),
                custom_scores[-1],
                np.mean(custom_scores),
                completions[-1],
                np.mean(completions),
                deadlocks[-1],
                np.mean(deadlocks),
                steps[-1],
                train_env._max_episode_steps,
                policy.choice_selector.get_parameter(),
                np.sum(choices_count),
                np.sum(num_exploration_choices),
                format_choices_probabilities(choices_probs)
            ), end="\n"
        )

        # Evaluate policy and log results at some interval
        # (always evaluate the final episode)
        if (args.training.eval_env.episodes > 0 and
            ((episode > 0 and episode % args.training.checkpoint == 0) or
             (episode == args.training.train_env.episodes))):
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
        tensorboard_log(writer, "training/steps", steps[-1], episode)
        tensorboard_log(
            writer, "training/choices_count",
            np.sum(choices_count), episode
        )
        tensorboard_log(
            writer, "training/exploration_choices",
            np.sum(num_exploration_choices), episode
        )
        tensorboard_log(
            writer, "training/exploration_prob",
            policy.choice_selector.get_parameter(), episode
        )
        tensorboard_log(
            writer, "training/loss",
            policy.loss.data.item(), episode
        )
        tensorboard_log(writer, "training/score", scores[-1], episode)
        tensorboard_log(
            writer, "training/custom_score",
            custom_scores[-1], episode
        )
        tensorboard_log(writer, "training/completion",
                        completions[-1], episode)
        tensorboard_log(
            writer, "training/buffer_size",
            len(policy.memory), episode
        )
        tensorboard_log(writer, "training/deadlocks", deadlocks[-1], episode)

        # Log training time info to tensorboard
        tensorboard_log(writer, "timer/reset", reset_timer.get(), episode)
        tensorboard_log(writer, "timer/step", step_timer.get(), episode)
        tensorboard_log(writer, "timer/learn", learn_timer.get(), episode)
        tensorboard_log(
            writer, "timer/total",
            training_timer.get_current(), episode
        )

    # Print final training info
    print("\n\r🏁 Training ended \tTrained {} trains on {}x{} grid for {} episodes \t Evaluated on {} episodes every {} episodes".format(
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
    if args.generic.enable_wandb:
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
    scores, custom_scores, completions, steps, choices_count, deadlocks = [], [], [], [], [], []

    # Do the specified number of episodes
    print('\nStarting validation:')
    for episode, seed in enumerate(eval_seeds):
        score, custom_score = 0.0, 0.0
        final_step = 0
        choices_taken = []

        # Reset environment and renderer
        if not args.training.eval_env.all_random:
            obs, info = env.reset(random_seed=seed)
        else:
            obs, info = env.reset(
                regenerate_rail=True, regenerate_schedule=True,
            )
        if args.training.renderer.evaluation and episode % args.training.renderer.eval_checkpoint == 0:
            env_renderer = env.get_renderer()

        # Compute agents with same source
        agents_with_same_start = env.get_agents_same_start()

        # Do an episode
        for step in range(env._max_episode_steps):

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
                            obs[agent], legal_choices, training=False
                        )
                        assert is_best == True
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
            obs, rewards, custom_rewards, done, info = env.step(action_dict)

            # Render an episode at some interval
            if args.training.renderer.evaluation and episode % args.training.renderer.eval_checkpoint == 0:
                env_renderer.render_env(
                    show=True, show_observations=False, show_predictions=True, show_rowcols=True
                )

            # Update agents scores
            for agent in env.get_agent_handles():
                score += rewards[agent]
                custom_score += custom_rewards[agent]

            # Break if every agent arrived
            final_step = step
            if done['__all__'] or env.check_if_all_blocked(info["deadlocks"]):
                break

        # Close window
        if args.training.renderer.evaluation and episode % args.training.renderer.eval_checkpoint == 0:
            env_renderer.close_window()

        # Save final scores
        scores.append(score / (env._max_episode_steps * env.get_num_agents()))
        custom_scores.append(custom_score / env.get_num_agents())
        completions.append(
            sum(done[idx] for idx in env.get_agent_handles()) / env.get_num_agents())
        steps.append(final_step)
        choices_count.append(len(choices_taken))
        deadlocks.append(
            sum(int(v) for v in info["deadlocks"].values()) / env.get_num_agents())

        # Print evaluation results on one episode
        print(
            '\r🚂 Validation {:3n}'
            '\t 🏆 Score: {:+5.4f}'
            '\t 🏅 Custom score: {:+5.4f}'
            '\t 💯 Done: {:7.2%}'
            '\t 💀 Deadlocks: {:7.2%}'
            '\t 🦶 Steps: {:4n}/{:4n}'
            '\t 🤔 Choices: {:4n}'.format(
                episode,
                scores[-1],
                custom_scores[-1],
                completions[-1],
                deadlocks[-1],
                steps[-1],
                env._max_episode_steps,
                choices_count[-1]
            ), end="\n"
        )

    # Print validation results
    print(
        '\r✅ Validation ended'
        '\t 🏆 Avg score: {:+5.2f}'
        '\t 🏅 Avg custom score: {:+5.2f}'
        '\t 💯 Avg done: {:7.2%}'
        '\t 💀 Avg deadlocks: {:7.2%}'
        '\t 🦶 Avg steps: {:5.2f}'
        '\t 🤔 Avg choices: {:5.2f}'.format(
            np.mean(scores),
            np.mean(custom_scores),
            np.mean(completions),
            np.mean(deadlocks),
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
        writer, "evaluation/custom_scores", custom_scores, train_episode,
        plot=['mean', 'std', 'hist']
    )
    tensorboard_log(
        writer, "evaluation/completions", completions, train_episode,
        plot=['mean', 'std', 'hist']
    )
    tensorboard_log(
        writer, "evaluation/deadlocks", deadlocks, train_episode,
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
    os.environ["WANDB_MODE"] = "dryrun"
    with open('parameters.yml', 'r') as conf:
        args = yaml.load(conf, Loader=yaml.FullLoader)
    if args["generic"]["enable_wandb"]:
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


if __name__ == "__main__":
    main()
