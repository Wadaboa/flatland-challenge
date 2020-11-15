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

## Model

* [ ] Insert prior bias to the network
* [ ] Initialize weights of the network
* [ ] Check hyperparameter network
* [ ] Change reward based on choices
* [ ] Implement other Networks

## Policy

* [ ] Finish ActionPolicy development
