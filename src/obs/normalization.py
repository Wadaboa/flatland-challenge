import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailAgentStatus

import utils


####################################################################
#################### Binary tree observation #######################
####################################################################

BT_LOWER, BT_UPPER = -1, 1
BT_UNDER, BT_OVER = -2, 2


def dumb_normalize_binary_tree_obs(observation):
    '''
    Substitute infinite values with a lower bound (e.g. -1),
    but avoid scaling observations
    '''
    normalized_observation = observation.copy()
    normalized_observation[normalized_observation == -np.inf] = BT_LOWER
    normalized_observation[normalized_observation == np.inf] = BT_LOWER
    return normalized_observation


def normalize_binary_tree_obs(observation, remaining_agents, max_malfunction, fixed_radius):
    '''
    Normalize the given observations by performing min-max scaling
    over individual features
    '''
    normalized_observation = observation.copy()
    num_agents = normalized_observation[:, :, 0:4]
    agent_distances = normalized_observation[:, :, 4:6]
    malfunctions = normalized_observation[:, :, 6:8]
    target_distances = normalized_observation[:, :, 8]
    turns_to_node = normalized_observation[:, :, 9]
    c_nodes = normalized_observation[:, :, 10]
    deadlocks = normalized_observation[:, :, 11]
    deadlock_distances = normalized_observation[:, :, 12]
    are_forks = normalized_observation[:, :, 13]
    stop_actions = normalized_observation[:, :, 14]

    # Normalize number of agents in path
    num_agents = utils.min_max_scaling(
        num_agents, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=remaining_agents
    )

    # Normalize malfunctions
    malfunctions = utils.min_max_scaling(
        malfunctions, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=max_malfunction
    )

    # Normalize common nodes
    c_nodes = utils.min_max_scaling(
        c_nodes, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=remaining_agents
    )

    # Normalize deadlocks
    deadlocks = utils.min_max_scaling(
        deadlocks, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=remaining_agents
    )

    # Normalize distances
    agent_distances = utils.min_max_scaling(
        agent_distances, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=-fixed_radius, known_max=fixed_radius
    )
    target_distances = utils.min_max_scaling(
        target_distances, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=fixed_radius
    )
    turns_to_node = utils.min_max_scaling(
        turns_to_node, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=fixed_radius
    )
    deadlock_distances = utils.min_max_scaling(
        deadlock_distances, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=-fixed_radius, known_max=fixed_radius
    )

    # Normalize stop actions
    stop_actions = utils.min_max_scaling(
        stop_actions, BT_LOWER, BT_UPPER, BT_LOWER, BT_UPPER,
        known_min=0, known_max=fixed_radius
    )

    # Build the normalized observation
    normalized_observation[:, :, 0:4] = num_agents
    normalized_observation[:, :, 4:6] = agent_distances
    normalized_observation[:, :, 6:8] = malfunctions
    normalized_observation[:, :, 8] = target_distances
    normalized_observation[:, :, 9] = turns_to_node
    normalized_observation[:, :, 10] = c_nodes
    normalized_observation[:, :, 11] = deadlocks
    normalized_observation[:, :, 12] = deadlock_distances
    normalized_observation[:, :, 13] = are_forks
    normalized_observation[:, :, 14] = stop_actions

    # Sanity check
    normalized_observation[normalized_observation == -np.inf] = BT_LOWER
    normalized_observation[normalized_observation == np.inf] = BT_UPPER
    normalized_observation = np.clip(
        normalized_observation, BT_LOWER, BT_UPPER
    )

    # Check if the output is in range [BT_LOWER, BT_UPPER]
    assert np.logical_and(
        normalized_observation >= BT_LOWER,
        normalized_observation <= BT_UPPER
    ).all(), (observation, normalized_observation)

    return normalized_observation


####################################################################
######################## Tree observation ##########################
####################################################################

T_CLIP_MIN, T_CLIP_MAX = -1, 1


def max_lt(seq, val):
    '''
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    '''
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    '''
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    '''
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    '''
    This function returns the difference between min and max value of an observation
    '''
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node):
    '''
    This function separates features of the given node into logical groups
    '''
    # Data features
    data = np.zeros(6)
    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    # Distance features
    distance = np.zeros(1)
    distance[0] = node.dist_min_to_target

    # Agent data features
    agent_data = np.zeros(4)
    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node, current_tree_depth, max_tree_depth):
    '''
    This function recursively extracts information starting from the given node
    '''
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return (
            [-np.inf] * num_remaining_nodes * 6,
            [-np.inf] * num_remaining_nodes,
            [-np.inf] * num_remaining_nodes * 4
        )

    data, distance, agent_data = _split_node_into_feature_groups(node)
    if not node.childs:
        return data, distance, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            node.childs[direction], current_tree_depth + 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def split_tree_into_feature_groups(tree, max_tree_depth):
    '''
    This function splits the tree into three difference arrays of values
    '''
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(
            tree.childs[direction], 1, max_tree_depth
        )
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def normalize_tree_obs(observation, tree_depth, radius):
    '''
    This function normalizes the observation used by the RL algorithm
    '''
    data, distance, agent_data = split_tree_into_feature_groups(
        observation, tree_depth
    )

    data = norm_obs_clip(
        data, clip_min=T_CLIP_MIN, clip_max=T_CLIP_MAX,
        fixed_radius=radius
    )
    distance = norm_obs_clip(
        distance, clip_min=T_CLIP_MIN, clip_max=T_CLIP_MAX,
        normalize_to_range=True
    )
    agent_data = np.clip(agent_data, T_CLIP_MIN, T_CLIP_MAX)
    normalized_obs = np.concatenate(
        (np.concatenate((data, distance)), agent_data)
    )
    return normalized_obs
