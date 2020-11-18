# :memo: TODOs

## Observation

* [ ] Test deadlock distance with variable speeds
* [ ] Test malfunctions observations
* [ ] Add malfunction turns to cumulative weights
* [ ] Check normalization correctness
* [ ] Handle observation not present at all
* [ ] Store agents in deadlock and substitute their shortest path as if they cannot reach their target (i.e. store current and next node)

## Main

* [ ] Add check for deadlock with agents on nodes
* [ ] Change score if episode is closed by all agents being in deadlock
* [ ] Add file logger to train.py

## Model

* [ ] Insert prior bias into the network (to replicate standard action probabilities)
* [x] Initialize weights of the network
* [ ] Check hyperparameter network
* [x] Change reward based on choices
* [ ] Implement other Networks

## Policy

* [ ] Finish and check ActionPolicy development
* [x] Review _get_q_targets_next

## Replay Buffer

* [ ] Prioritized experience replay
* [ ] Sample by giving more weight to the latest inserted experiences
* [ ] Sample experiences of the same agent
* [ ] Sample experiences in temporal order (and add a recurrent unit to the DQN)
* [ ] Stop storing legal_choices into the buffer (we never use them)
