import gymnasium as gym
import numpy as np
from controller import Supervisor


class RobotSupervisorEnv(Supervisor, gym.Env):

    def __init__(self):
        super().__init__()
        self.timestep = 256
        self.steps = 0
        self.reward = 0
        self.terminated = False
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        # obs = super().reset(seed=seed)
        self.steps = 0
        self.reward = 0
        self.episode_count += 1
        self.terminated = False
        self.truncated = False
        print("Episode: ", self.episode_count)
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)
        # self.simulationReset()
        # self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_default_observation(), self.get_info()

    def step(self, action):
        self.steps += 1
        self.apply_action(action)
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
        obs = self.get_observations()
        reward = self.get_reward(obs)
        if action == 1 or 2:
            reward -= 10
        terminated = self.is_done(obs)
        truncated = self.truncated
        info = self.get_info()
        return obs, reward, terminated, truncated, info


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