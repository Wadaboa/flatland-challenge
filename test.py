import os
import time
from datetime import datetime

import numpy as np
import yaml
from tabulate import tabulate

from flatland.envs.rail_env import RailEnvActions, RailAgentStatus

import utils
import env_utils
from policies import POLICIES
from action_selectors import PARAMETER_DECAYS, ACTION_SELECTORS


def print_agents_info(env):
    '''
    Print information for each agent in a specific step
    '''
    _status_table = []
    for handle, agent in enumerate(env.agents):
        _status_table.append([
            handle,
            agent.status,
            agent.speed_data["speed"],
            agent.speed_data['position_fraction'],
            (
                agent.initial_position[0],
                agent.initial_position[1],
                agent.direction
            ) if agent.status == RailAgentStatus.READY_TO_DEPART else
            (
                agent.position[0],
                agent.position[1],
                agent.direction
            ) if agent.status != RailAgentStatus.DONE_REMOVED else (
                'DONE'
            ),
            agent.target,
            agent.malfunction_data['malfunction']
        ])
    print(tabulate(
        _status_table,
        [
            "Handle", "Status", "Speed", "Position fraction",
            "Position", "Target", "Malfunction"
        ],
        colalign=["center"] * 7
    ))


def test_agents(args):
    '''
    Test agents on the specified environment
    '''
    # Initialize threads and seeds
    utils.set_num_threads(args.generic.num_threads)
    if args.generic.fix_random:
        utils.fix_random(args.generic.random_seed)

    # Create railway environment and renderer
    env = env_utils.create_rail_env(args, load_env=args.testing.load)
    obs, info = env.reset(random_seed=args.env.seed)
    env_renderer = env.get_renderer()

    # Load the model if provided
    if args.testing.model:
        parameter_decay = PARAMETER_DECAYS["none"](
            parameter_start=args.parameter_decay.start
        )
        action_selector = ACTION_SELECTORS["greedy"](parameter_decay)
        policy_type = args.policy.type.get_true_key()
        policy = POLICIES[policy_type](
            args, env.state_size, action_selector, training=False
        )
        policy.load(args.testing.model)
    else:
        policy = POLICIES["random"]()

    # Print agents tasks
    _tasks_table = []
    for handle, agent in enumerate(env.agents):
        _tasks_table.append([
            handle,
            agent.status,
            agent.speed_data["speed"],
            (
                agent.initial_position[0],
                agent.initial_position[1],
                agent.direction
            ),
            agent.target
        ])
    print(tabulate(
        _tasks_table,
        ["Handle", "Status", "Speed", "Source", "Target"],
        colalign=["center"] * 5
    ))
    print()

    # Create frames directory
    now = datetime.now()
    test_id = now.strftime('%Y%m%d-%H%M%S')
    if args.testing.save_frames:
        frames_dir = f"tmp/frames/{test_id}"
        os.makedirs(frames_dir, exist_ok=True)

    # Compute agents with same source
    agents_with_same_start = env.get_agents_same_start()

    action_dict = dict()
    legal_choices = dict()
    score, custom_score = 0.0, 0.0
    for step in range(args.env.max_moves + 1):
        print(f"Iteration {step}")

        # Prioritize entry of faster agent in the environment
        for position in agents_with_same_start:
            if len(agents_with_same_start[position]) > 0:
                del agents_with_same_start[position][0]
                for agent in agents_with_same_start[position]:
                    info['action_required'][agent] = False

        # Choose an action for each agent in the environment
        for agent in range(env.get_num_agents()):
            action = RailEnvActions.DO_NOTHING.value
            if info['action_required'][agent]:
                if env.railway_encoding.is_real_decision(agent):
                    legal_actions = env.railway_encoding.get_agent_actions(
                        agent
                    )
                    legal_choices[agent] = env.railway_encoding.get_legal_choices(
                        agent, legal_actions
                    )
                    choice, is_best = policy.act(
                        obs[agent], legal_choices[agent], training=False
                    )
                    assert is_best == True
                    action = env.railway_encoding.map_choice_to_action(
                        choice, legal_actions
                    )
                    assert action != RailEnvActions.DO_NOTHING.value, (
                        choice, legal_actions
                    )
                    print(
                        f'Handle: {agent} - Choice {choice} - Action {action} - Legal choices {legal_choices[agent]}'
                    )
                else:
                    actions = env.railway_encoding.get_agent_actions(
                        agent
                    )
                    assert len(actions) == 1, actions
                    action = actions[0]
            action_dict.update({agent: action})

        # Perform the computed action
        obs, rewards, custom_rewards, done, info = env.step(action_dict)
        env_renderer.render_env(
            show=True, show_observations=False, show_predictions=True, show_rowcols=True
        )
        if args.testing.sleep > 0:
            time.sleep(args.testing.sleep)

        # Save renderer frame
        if args.testing.save_frames:
            env_renderer.gl.save_image(
                "{:s}/{:04d}.png".format(frames_dir, step)
            )

        # Update agents score
        for handle in range(env.get_num_agents()):
            score += rewards[handle]
            custom_score += custom_rewards[handle]

        # Print statistics
        normalized_score = (
            score / (args.env.max_moves * env.get_num_agents())
        )
        normalized_custom_score = custom_score / env.get_num_agents()
        print(
            f"Score: {round(normalized_score, 4)} / "
            f"Custom score: {round(normalized_custom_score, 4)}"
        )
        print_agents_info(env)
        print()

        # Check if every agent is arrived
        if done['__all__'] or env.check_if_all_blocked(info["deadlocks"]):
            break


def main():
    '''
    Test environment with the given model
    '''
    with open('parameters.yml', 'r') as conf:
        args = yaml.load(conf, Loader=yaml.FullLoader)
    args = utils.Struct(**args)
    test_agents(args)


if __name__ == "__main__":
    main()
