from flatland.envs.rail_env import RailEnvActions
from flatland.envs.persistence import RailEnvPersister


def agent_action(self, original_dir, final_dir):
    '''
    Return the action performed by an agent, by analyzing
    the starting direction and the final direction of the movement
    '''
    value = (final_dir.value - original_dir.value) % 4
    if value in (1, -3):
        return RailEnvActions.MOVE_RIGHT
    elif value in (-1, 3):
        return RailEnvActions.MOVE_LEFT
    return RailEnvActions.MOVE_FORWARD


def create_save_env(path, width, height, num_trains, max_cities,
                    max_rails_between_cities, max_rails_in_cities, grid=False, seed=0):
	'''
	Create a RailEnv environment with the given settings and save it as pickle
	'''
    rail_generator = sparse_rail_generator(
        max_num_cities=max_cities,
        seed=seed,
        grid_mode=grid,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_cities,
    )
    env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        number_of_agents=num_trains
    )
    save_env(path, env)


def save_env(path, env):
	'''
	Save the given RailEnv environment as pickle
	'''
    filename = os.path.join(
        path,
        f"{env.width}x{env.height}-{env.random_seed}.pkl"
    )
    RailEnvPersister.save(env, filename)


def get_seed(env, seed=None):
    '''
    Exploit the RailEnv to get a random seed
    '''
    return env._seed(seed)[0]
