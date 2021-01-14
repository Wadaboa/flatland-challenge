# Flatland challenge

![Example](assets/img/env.gif)

The Flatland challenge is a competition hosted by [AIcrowd](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/), where participants should tackle a multi agent reinforcement learning problem on trains. This repository contains solutions and approaches to the challenge hosted in 2020 and sponsored by NeurIPS.

Neural models have been implemented using the PyTorch framework and training results have been logged to [Weights & Biases](https://wandb.ai/) (also called `wandb`). Hyperparameters are handled via the use of a custom [parameters.yml](parameters.yml) file, that is accessed in almost every Python module of the project.

Our solutions mostly focus on implementing custom predictors and observators. Moreover, we tried to exploit both common models, like DQN, but also custom-made ones, like those based on GNNs. If you want to know more about our work, you can read the full [report](report/report.pdf).

## Installation

Before installing the actual Python libraries, make sure that you have installed the system packages listed below.

**Ubuntu/Debian**
```bash
sudo apt install libcairo2-dev pkg-config python3-dev libbz2-dev liblzma-dev
```
------

Next, to install all the libraries required by the Python package, run the following command:
```bash
chmod +x init/init.sh 
./init/init.sh <env> <device>
```

Here, `<env>` could be `pip` or `conda`, to choose what type of environment manager you want to use, while `<device>` could be `cpu` or `gpu`, to select the right dependencies based on the resources available.

If you select `conda` as the environment manager, you should have it installed on your system (see [here](https://docs.conda.io/en/latest/miniconda.html) for installation instructions).

Beware that the `pip` environment was only tested with `Python 3.6.3` (you can use [pyenv](https://github.com/pyenv/pyenv) to handle multiple Python versions), while the `conda` environment automatically installs `Python 3.6.8`. 

## Training

### New training

If you want to train one of the implemented models, make sure to select the required parameters in the `parameters.yml` file and then run `python3 src/train.py`. If everything goes as expected, you will find a text like the following on your standard output:

```
💾 Replay buffer status: 0/100000 experiences

🚉 Starting training     Training 7 trains on 48x27 grid for 5000 episodes      Evaluating on 20 episodes every 200 episodes

🧠 Model with training id 20201218-185537

🚂 Episode    0  🏆 Score: -0.1945 Avg: -0.1945  🏅 Custom score: -0.8782 Avg: -0.8782   💯 Done: 14.29%  Avg:  14.29%   💀 Deadlocks: 85.71%  Avg:  85.71%      🦶 Steps:  130/ 611     🎲 Exploration prob: 1.000    🤔 Choices:  156        🤠 Exploration:  42     🔀 Choices probs: ← 23.70% → 8.30% ◼ 17.90% 

...
```

One thing to note is that `wandb` logging should be disabled, since it requires a private access token linked to my personal account. Anyways, you can check the learning progress offline using Tensorboard, with the following command:

```bash
tensorboard --logdir="./runs" --port 6006
```

To view the Tensorboard interface, just open the link http://localhost:6006 on your browser.

### Previous training results

If you want to check results obtained by training models with specific set of parameters, you can visit the [flatland-challenge](https://wandb.ai/wadaboa/flatland-challenge?workspace=user-wadaboa) project in my wandb account. Each run has a "self-explanatory" name and contains all the parameters used to train the model, charts depicting the learning progress, `PyTorch`'s `.pt` model files and actual logs extracted from the standard output.

## Testing

If you want to test one pre-trained model, make sure to insert the model's file path (without the final extension) in `parameters.yml/testing/model` and adjust the other parameters in `parameters.yml` so that they are compatible with the ones used for training the loaded model. Then, simply run `python3 scr/test.py`. If everything goes as expected, you will find a text like the following on your standard output:

```
🚉 Starting testing      Testing 7 trains on 48x27 grid for 1 episodes

🚂 Test    0     🏆 Score: -1.0000 Avg: -1.0000  🏅 Custom score: -2.1365 Avg: -2.1365   💯 Done: 0.00%   Avg:   0.00%   💀 Deadlocks: 57.14%  Avg:  57.14%      🦶 Steps:  610/ 611     🤔 Choices:  306

...
```
