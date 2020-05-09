import numpy as np
import time

from flatland.core.env_prediction_builder import PredictionBuilder
# from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


random_seed = 14

rail_generator = sparse_rail_generator(
    max_num_cities=2,
    seed=random_seed,
    grid_mode=True,
    max_rails_between_cities=3,
    max_rails_in_city=1,
)


env = RailEnv(
    width=32,
    height=32,
    rail_generator=rail_generator,
    number_of_agents=1,
    obs_builder_object=observation_builder,
    remove_agents_at_target=True
)
env.reset()

env_renderer = RenderTool(
    env, gl="PILSVG",
    agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
    show_debug=True,
    screen_height=1000,
    screen_width=1000
)
env_renderer.render_env(show=True)
time.sleep(2)
