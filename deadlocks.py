from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import MOVEMENT_ARRAY, get_new_position


class DeadlocksDetector:
    '''
    Class containing code to track deadlocks during an episode,
    based on https://github.com/AlessandroLombardi/FlatlandChallenge
    '''

    def __init__(self):
        self.deadlocks = dict()
        self.deadlock_turns = dict()

    def reset(self, num_agents):
        '''
        Reset deadlock counters
        '''
        self.deadlocks = {a: False for a in range(num_agents)}
        self.deadlock_turns = {a: None for a in range(num_agents)}

    def step(self, env):
        '''
        Check for new deadlocks, updates counter and returns it
        '''
        agents = []
        for a in range(env.get_num_agents()):
            if env.agents[a].status == RailAgentStatus.ACTIVE:
                agents.append(a)
                if not self.deadlocks[a]:
                    self.deadlocks[a] = self._check_deadlocks(
                        agents, self.deadlocks, env
                    )
                if not self.deadlocks[a]:
                    del agents[-1]
                elif self.deadlock_turns[a] is None:
                    self.deadlock_turns[a] = env._elapsed_steps - 1
            else:
                self.deadlocks[a] = False

        return self.deadlocks, self.deadlock_turns

    def _check_feasible_transitions(self, pos, env):
        '''
        Function used to collect chains of blocked agents
        '''
        transitions = env.rail.get_transitions(*pos)
        n_transitions = 0
        occupied = 0
        agent_in_path = None
        for direction, values in enumerate(MOVEMENT_ARRAY):
            if transitions[direction] == 1:
                n_transitions += 1
                new_position = get_new_position(pos, direction)
                for agent in range(env.get_num_agents()):
                    if env.agents[agent].position == new_position:
                        occupied += 1
                        agent_in_path = agent
        if n_transitions > occupied:
            return None
        return agent_in_path

    def _check_next_pos(self, agent, env):
        '''
        Check the next pos and the possible transitions of an agent to find deadlocks
        '''
        pos = (*env.agents[agent].position, env.agents[agent].direction)
        return self._check_feasible_transitions(pos, env)

    def _check_deadlocks(self, agents, deadlocks, env):
        '''
        Recursive procedure to find out whether agents in `agents` are in a deadlock
        '''
        other_agent = self._check_next_pos(agents[-1], env)

        # No agents in front
        if other_agent is None:
            return False

        # Deadlocked agent in front or loop chain found
        if deadlocks[other_agent] or other_agent in agents:
            return True

        # Investigate further
        agents.append(other_agent)
        deadlocks[other_agent] = self._check_deadlocks(agents, deadlocks, env)

        # If the agent `other_agent` is in deadlock
        # also the last one in `agents` is
        if deadlocks[other_agent]:
            return True

        # Back to previous recursive call
        del agents[-1]
        return False
