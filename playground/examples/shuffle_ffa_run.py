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

import random
import pommerman
from pommerman import agents

def create_agents():
    return [
        agents.SimpleAgent(),
        agents.RandomRunAgent(),
        agents.RandomRunAgent(),
        agents.RandomRunAgent(),
        # agents.RandomAgent(),
        # agents.RandomAgent(),
        # agents.RandomAgent(),
        # agents.SimpleAgent(),
        # agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = create_agents()
    # Make the "Free-For-All" environment using the agent list
    # env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    total_reward = [0 for _ in agent_list]
    steps = 0
    num_episodes = 100
    winners_count = {i: 0 for i in range(len(agent_list))}

    for i_episode in range(num_episodes):
        shuffled_indices = random.sample(
            list(range(len(agent_list))), len(agent_list))
        agent_list = create_agents()
        shuffled_agents = [agent_list[i] for i in shuffled_indices]
        # print('shuffled_indices: ', shuffled_indices)
        # print('shuffled_agents: ', shuffled_agents)
        env = pommerman.make('PommeFFACompetition-v0', shuffled_agents)
        # env.set_agents(shuffled_agents)

        state = env.reset()
        done = False
        shuffled_reward = [0 for _ in shuffled_agents]
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            # Update total rewards.
            steps += 1
            for i, r in enumerate(reward):
                shuffled_reward[i] += r

        # winners = [shuffled_indices[w] for w in info.get('winners', [])]
        if i_episode < 100:
            print('Episode {} finished with info: {}'.format(i_episode, info))

        for winner in info.get('winners', []):
            winners_count[shuffled_indices[winner]] += 1

        # Episode reward normalization
        # total positive reward = 1, total negative reward = 1
        neg_total = sum(abs(r) for r in shuffled_reward if r < 0)
        shuffled_reward = [r if r > 0 else float(r) / neg_total for r in shuffled_reward]

        for (i, r) in enumerate(shuffled_reward):
            total_reward[shuffled_indices[i]] += r

    reward_episode = [float(r) / num_episodes for r in total_reward]
    print('avg_episode_reward: ', reward_episode)
    print('winners count: ', winners_count)

    env.close()


if __name__ == '__main__':
    main()
