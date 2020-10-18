import os
import time
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
from tabulate import tabulate

from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from observations import CustomObservation
from predictions import ShortestPathPredictor
from policies import RandomPolicy


RANDOM_SEED = 42


def parse_args():
    '''
    Parse terminal arguments
    '''
    parser = ArgumentParser()
    parser.add_argument(
        "--test_env", action='store', default="",
        help="path to the test environment file", type=str
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
        "--enable_renderer", action='store_true', default=True,
        help="enable renderer"
    )
    parser.add_argument(
        "--max_moves", action='store', default=500,
        help="maximum number of moves in an episode", type=int
    )
    parser.add_argument(
        "--sleep", action='store', default=2,
        help="seconds to sleep between moves", type=int
    )
    parser.add_argument(
        "--save_frames", action='store_true', default=True,
        help="save intermediate renderer frames"
    )
    parser.add_argument(
        "--max_depth", action='store', default=20,
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
    return parser.parse_args()


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
                agent.position[0],
                agent.position[1],
                agent.direction
            ),
            agent.target,
            agent.malfunction_data['malfunction']
        ])
    print(tabulate(
        _status_table,
        [
            "Handle", "Status", "Speed", "Position fraction",
            "Source", "Target", "Malfunction"
        ],
        colalign=["center"] * 7
    ))


def main():
    '''
    Test environment with custom observation and prediction
    '''
    args = parse_args()
    malfunctions = None
    if args.malfunction:
        malfunctions = ParamMalfunctionGen(MalfunctionParameters(
            malfunction_rate=args.malfunction_rate,
            min_duration=args.malfunction_min_duration,
            max_duration=args.malfunction_max_duration
        ))

    if args.test_env:
        rail_generator = rail_from_file(args.test_env)
    else:
        rail_generator = sparse_rail_generator(
            max_num_cities=args.max_cities,
            seed=RANDOM_SEED,
            grid_mode=args.grid,
            max_rails_between_cities=args.max_rails_between_cities,
            max_rails_in_city=args.max_rails_in_cities,
        )
    predictor = ShortestPathPredictor(max_depth=args.max_depth)
    observation_builder = CustomObservation(
        max_depth=args.max_depth, predictor=predictor
    )
    if args.variable_speed:
        speed_map = {
            1.: 0.25,
            1. / 2.: 0.25,
            1. / 3.: 0.25,
            1. / 4.: 0.25
        }
    else:
        speed_map = {1.: 1}
    schedule_generator = sparse_schedule_generator(speed_map)

    # Construct the environment
    env = RailEnv(
        width=args.width,
        height=args.height,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=args.num_trains,
        obs_builder_object=observation_builder,
        malfunction_generator=malfunctions,
        remove_agents_at_target=True
    )
    observations, _ = env.reset()

    # Initiate the renderer
    if args.enable_renderer:
        env_renderer = RenderTool(
            env,
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            screen_height=600,
            screen_width=800
        )
        env_renderer.render_env(show=True)

    # Initialize the agent
    controller = RandomPolicy()

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

    # Compute agents with same source
    agents_with_same_start = set()
    for handle_one, agent_one in enumerate(env.agents):
        for handle_two, agent_two in enumerate(env.agents):
            if handle_one != handle_two and agent_one.initial_position == agent_two.initial_position:
                agents_with_same_start.add(handle_one)

    # Create frames directory
    now = datetime.now()
    training_id = now.strftime('%Y%m%d-%H%M%S')
    if args.save_frames:
        frames_dir = f"tmp/frames/{training_id}"
        os.makedirs(frames_dir, exist_ok=True)

    # Initialize the action dictionary
    action_dict = dict()
    for handle in range(env.get_num_agents()):
        action_dict[handle] = 1

    score = 0.0
    for step in range(args.max_moves):
        print(f"Iteration {step}")

        # Choose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = controller.act(observations[a])
            action_dict.update({a: action})

        # Perform the computed action
        next_obs, all_rewards, done, info = env.step(action_dict)
        if args.enable_renderer:
            env_renderer.render_env(
                show=True, show_observations=False, show_predictions=True
            )
        time.sleep(args.sleep)

        # Save renderer frame
        if args.save_frames:
            env_renderer.gl.save_image(
                "{:s}/{:04d}.png".format(frames_dir, step)
            )

        # Update replay buffer and train agent
        for handle in range(env.get_num_agents()):
            controller.step((
                observations[handle],
                action_dict[handle],
                all_rewards[handle],
                next_obs[handle],
                done[handle]
            ))
            score += all_rewards[handle]

        # Save observations and check if every agent is arrived
        observations = next_obs.copy()
        if done['__all__']:
            break

        # Print statistics
        print(f"Score: {score}")
        print_agents_info(env)
        print(next_obs)
        print()


if __name__ == "__main__":
    main()
