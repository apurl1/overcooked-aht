class BaseScriptPeriod:
    """A base class for script period."""

    def __init__(self, period_name):
        self.period_name = period_name

    def reset(self, obs):
        """reset some script period"""
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError

    def done(self, obs):
        raise NotImplementedError


class BaseScriptAgent:
    """A script agent consists of several script periods."""

    def __init__(self):
        pass

    def reset(self, obs):
        """reset state"""

    def step(self, obs):
        raise NotImplementedError
