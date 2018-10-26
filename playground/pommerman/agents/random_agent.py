'''An agent that preforms a random action each step'''
from . import BaseAgent
from .. import constants


class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return action_space.sample()


class RandomRunAgent(BaseAgent):
    """The Random Run Agent that runs to a random direction given an
    action_space.
    """

    def act(self, obs, action_space):
        action = action_space.sample()
        while action == constants.Action.Bomb.value:
            action = action_space.sample()
        return action
