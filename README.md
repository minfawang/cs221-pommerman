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

#### Run game

```bash
cd playground

# Run this command every time before executing the programs below.
# "develop" vs "install": https://stackoverflow.com/a/19048754/4115411
python setup.py develop

# Control agents programmatically
python examples/simple_ffa_run.py

# Control agents from CLI
# See more detailed docs at: playground/docs/CLI.md
pom_battle --agents=player::arrows,test::agents.SimpleAgent,random::null,random::null --times=2
```

#### Run Training

All the learning scripts are under 
```bash
playground/examples/tf_*.py
```

To start the training, modify the corresponding flags within the .py file, and run with python directly within the virtual environment.


# Submission

http://web.stanford.edu/class/cs221/project.html#p-proposal

# Resources

From official [research.md](https://github.com/MultiAgentLearning/playground/blob/master/docs/research.md):

1. Proximal Policy Optimization (PPO) [14 Refs, 264 Cites] [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
2. Multi-Agent DDPG [36 Refs, 86 Cites] [https://github.com/openai/maddpg](https://github.com/openai/maddpg) Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments https://arxiv.org/abs/1706.02275
3. Monte Carlo Tree Search [https://gnunet.org/sites/default/files/Browne%20et%20al%20-%20A%20survey%20of%20MCTS%20methods.pdf](https://gnunet.org/sites/default/files/Browne%20et%20al%20-%20A%20survey%20of%20MCTS%20methods.pdf)
4. Monte Carlo Tree Search and Reinforcement Learning [https://www.jair.org/media/5507/live-5507-10333-jair.pdf](https://www.jair.org/media/5507/live-5507-10333-jair.pdf)
5. Cooperative Multi-Agent Learning [https://link.springer.com/article/10.1007/s10458-005-2631-2](https://link.springer.com/article/10.1007/s10458-005-2631-2)
6. Opponent Modeling in Deep Reinforcement Learning [25 Refs, 27 Cites] [http://www.umiacs.umd.edu/~hal/docs/daume16opponent.pdf](http://www.umiacs.umd.edu/~hal/docs/daume16opponent.pdf) https://arxiv.org/abs/1609.05559
7. Machine Theory of Mind [67 Refs, 7 Cites] [https://arxiv.org/pdf/1802.07740.pdf](https://arxiv.org/pdf/1802.07740.pdf)
8. Coordinated Multi-Agent Imitation Learning [https://arxiv.org/pdf/1703.03121.pdf](https://arxiv.org/pdf/1703.03121.pdf)
9. Deep Reinforcement Learning from Self-Play in Imperfect-Information Games [https://arxiv.org/pdf/1603.01121.pdf](https://arxiv.org/pdf/1603.01121.pdf) and[http://proceedings.mlr.press/v37/heinrich15.pdf](http://proceedings.mlr.press/v37/heinrich15.pdf)
10. Autonomous Agents Modelling Other Agents [250 Refs, 7 Cites] [http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AIJ18-Albrecht.pdf](http://www.cs.utexas.edu/~pstone/Papers/bib2html-links/AIJ18-Albrecht.pdf)
