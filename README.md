# cs221-pommerman
RL agent for Pommerman: https://www.pommerman.com/

# Setup

#### Prerequisites

* **Python3**: we use Python version 3+ for this project.
* [Pipenv](https://github.com/pypa/pipenv): Python package manager and virtual environment. Can be installed with command `pip install pipenv`.

#### Initial Setup

At the first time, run the following commands:

```bash
git clone git@github.com:minfawang/cs221-pommerman.git  # Clones repo.
cd cs221-pommerman  # Changes your directory to the root of the repo.
# If you use a conda custom Python binary, then you may use the
# command in the comment below:
# pipenv --python /usr/local/bin/python3 install
pipenv --three install  # Create a virtual env using Python3.

# Enter virtual env.
pipenv shell

# Set up custom python kernel with correct binary and dependency.
# https://stackoverflow.com/a/47296960
python -m ipykernel install --user --name=cs221-pommerman
```
