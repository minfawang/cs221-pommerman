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
from pommerman import agents


def main():
    '''Simple function to bootstrap a game.

       Use this as an example to set up your training env.
    '''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        # agents.RandomRunAgent(),
        # agents.RandomRunAgent(),
        # agents.RandomRunAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        # agents.SimpleAgent(),
        # agents.RandomAgent(),
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    total_reward = [0 for _ in agent_list]
    steps = 0
    num_episodes = 10
    winners_count = {i: 0 for i in range(len(agent_list))}

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            # Update total rewards.
            steps += 1
            for i, r in enumerate(reward):
                total_reward[i] += r

        for winner in info.get('winners', []):
            winners_count[winner] += 1

        if i_episode < 10:
            print('Episode {} finished with info: {}'.format(i_episode, info))

    reward_episode = [float(r) / num_episodes for r in total_reward]
    print('avg_episode_reward: ', reward_episode)
    print('winners count: ', winners_count)

    env.close()


if __name__ == '__main__':
    main()
