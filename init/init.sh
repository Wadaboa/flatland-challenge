#!/bin/bash

# Use pip as environment/package manager
if [ $1 == "pip" ]; then

    # Store root path
    PATH=$(dirname $(dirname "$(realpath $0)" ) )
    ENVIRONMENT_PATH="venv"

    # Create the virtual environment and activate it
    python3 -m venv $PATH/$ENVIRONMENT_PATH
    . "$PATH"/"$ENVIRONMENT_PATH"/bin/activate

    # Upgrade pip, setuptools and wheel
    pip install pip==20.2.3
    pip install --upgrade setuptools wheel

    # Install dependencies
    pip install -r requirements.txt

    # Install PyTorch and PyTorch Geometric
    if [ $2 == "cpu" ]; then
        pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.6.0+cpu
        pip install -r torch_geometric_cpu.txt
    elif [ $2 == "gpu" ]; then
        pip install torch==1.6.0
        pip install -r torch_geometric_gpu.txt
    fi

    # Install Flatland
    pip install flatland-rl==2.2.2

# Use conda as environment/package manager
elif [ $1 == "conda" ]; then
    
    # Create and activate conda environment
    conda env create --name flatland-rl -f environment.yml
    conda activate flatland-rl
    
    # Install PyTorch and PyTorch Geometric
    if [ $2 == "cpu" ]; then
        conda install pytorch==1.6.0 cpuonly -c pytorch
        $(which pip) install -r torch_geometric_cpu.txt
    elif [ $2 == "gpu" ]; then
        conda install pytorch==1.6.0 cudatoolkit=10.2 -c pytorch
        $(which pip) install -r torch_geometric_gpu.txt
    fi

fi
