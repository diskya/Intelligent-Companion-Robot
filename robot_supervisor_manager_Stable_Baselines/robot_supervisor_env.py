import gym
import numpy as np
from controller import Supervisor


class RobotSupervisorEnv(Supervisor, gym.Env):

    def __init__(self):
        super().__init__()
        self.timestep = 32

    def reset(self):
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_default_observation()

    def step(self, action):
        self.apply_action(action)
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    def apply_action(self, action):
        raise NotImplementedError

    def get_default_observation(self):
        raise NotImplementedError

    def get_observations(self):
        raise NotImplementedError

    def get_reward(self, action):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError
    
    def normalize_to_range(self, value, min_val, max_val, new_min, new_max, clip=False):
        value = float(value)
        min_val = float(min_val)
        max_val = float(max_val)
        new_min = float(new_min)
        new_max = float(new_max)
        if clip:
            return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
        else:
            return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max