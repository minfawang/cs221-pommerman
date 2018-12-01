'''An example to show how to set up an pommerman game programmatically

Explanation of rewards:
# Suppose we have 4 agents, below may be a sequence of rewards after each step.
# Essentially the reward of an agent is: 0 if alive, -1 if dead, and +1 if
# you are the only one alive.
# reward = [0, 0, 0, 0]
# reward = [0, 0, 0, 0]
# reward = [0, 0, 0, -1]
# reward = [0, 0, -1, -1]
# reward = [-1, 1, -1, -1]
'''
import pommerman
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


DEBUG = True
SHOULD_RENDER = True
NUM_EPISODES = 10
MAX_EPISODE_TIMTESTAMPS = 2000
MODEL_DIR = '/Users/voiceup/Git/cs221-pommerman/playground/notebooks/saved_ckpts/'
REPORT_EVERY_ITER = 100
SAVE_EVERY_ITER = 1000

def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def featurize(obs):
    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])

    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    return np.concatenate((
        board, bomb_blast_strength, bomb_life, position, ammo,
        blast_strength, can_kick, teammate, enemies))


class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


def create_env_agent():
  # Instantiate the environment
  config = ffa_v0_fast_env()
  env = Pomme(**config["env_kwargs"])
  env.seed(0)

  # Create a Proximal Policy Optimization agent
  agent = PPOAgent(
      states=dict(type='float', shape=env.observation_space.shape),
      actions=dict(type='int', num_actions=env.action_space.n),
      network=[
          dict(type='dense', size=64),
          dict(type='dense', size=64)
      ],
      batching_capacity=1000,
      step_optimizer=dict(
          type='adam',
          learning_rate=1e-4
      ),

      # PGModel
      baseline_mode='network',
      baseline=dict(type='custom', network=[
          dict(type='dense', size=64),
          dict(type='dense', size=64)
      ]),
      baseline_optimizer=dict(
          type='adam',
          learning_rate=1e-4
      ),
  )

  # Add 3 random agents
  agents = []
  for agent_id in range(3):
      agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

  # Add TensorforceAgent
  agent_id += 1
  agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
  env.set_agents(agents)
  env.set_training_agent(agents[-1].agent_id)
  env.set_init_game_state(None)

  return (env, agent)


class WrappedEnv(OpenAIGym):
    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]

        if DEBUG:
          print('agent_state, terminal, agent_reward: ', agent_state, terminal, agent_reward)
          input('\n press any key to step forward \n')

        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs


# Callback function printing episode statistics
def episode_finished(r):
  if r.episode % REPORT_EVERY_ITER == 0:
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(
          ep=r.episode, ts=r.episode_timestep, reward=r.episode_rewards[-1]))

  if r.episode % SAVE_EVERY_ITER == 0:
    r.agent.save_model(MODEL_DIR)

  return True


def main():
  # Print all possible environments in the Pommerman registry
  print(pommerman.REGISTRY)

  # Create a set of agents (exactly four)
  env, agent = create_env_agent()

  wrapped_env = WrappedEnv(env, visualize=SHOULD_RENDER)
  runner = Runner(agent=agent, environment=wrapped_env)
  runner.run(
      episodes=NUM_EPISODES,
      max_episode_timesteps=MAX_EPISODE_TIMTESTAMPS,
      episode_finished=episode_finished
  )

  try:
      runner.close()
  except AttributeError as e:
      print('AttributeError: ', e)


if __name__ == '__main__':
    main()
