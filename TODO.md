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

* [ ] Insert prior bias to the network
* [ ] Initialize weights of the network
* [ ] Check hyperparameter network
* [x] Change reward based on choices
* [ ] Implement other Networks

## Policy

* [ ] Finish and check ActionPolicy development
* [x] Review _get_q_targets_next

## Replay Buffer

* [ ] Try Prioritized Experience Replay
* [ ] Stop storing legal_choices into the Buffer - we never use them
