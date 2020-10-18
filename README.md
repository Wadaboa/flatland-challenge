# flatland-challenge

Multi Agent Reinforcement Learning on Trains.

## Installation

### Anaconda

Install [Anaconda](https://www.anaconda.com/distribution/) and create a new conda environment:

```bash
conda env create --name flatland-rl -f environment.yml
conda activate flatland-rl
```

### Pip

Make sure that you have `Python 3.6` (the project has been tested both with `Python 3.6.3` and `Python 3.6.8`) installed on your system. Then, `cd` in the root folder of this project and run the following command:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The command will create a virtual environment (named `venv`), activate it and install all the necessary dependencies.

### SVG rendering

#### MacOS with pyenv and Homebrew

If you intend to use SVG rendering as the flatland GUI, you may have to do the following in a `macOS` environment, with [pyenv](https://github.com/pyenv/pyenv) and [Homebrew](https://brew.sh/index_it) already installed:

```bash
brew install tcl-tk
```

```bash
env \
  PATH="$(brew --prefix tcl-tk)/bin:$PATH" \
  LDFLAGS="-L$(brew --prefix tcl-tk)/lib" \
  CPPFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PKG_CONFIG_PATH="$(brew --prefix tcl-tk)/lib/pkgconfig" \
  CFLAGS="-I$(brew --prefix tcl-tk)/include" \
  PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I$(brew --prefix tcl-tk)/include' --with-tcltk-libs='-L$(brew --prefix tcl-tk)/lib -ltcl8.6 -ltk8.6'" \
  pyenv install 3.8.1
```

These commands will install the latest `TK` version and bind its sources to the Python build. In this example, `Python 3.8.1` was used, along with `flatland-rl 2.1.10` (since the latest `flatland-rl 2.2.2` has a bug with SVG rendering).
