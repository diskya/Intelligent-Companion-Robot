from robot_supervisor_env import RobotSupervisorEnv
from gymnasium import spaces
from agent_controller import Supervisor, Robot

import numpy as np


class HomeRobotEnv(RobotSupervisorEnv):

    def __init__(self):

        super().__init__()
        self.TIME_STEP = 512
        # Set up gym spaces
        self.observation_space = spaces.Dict(
            {
                "robot_position": spaces.Box(-10, 0, shape=(2,), dtype=float),
                "target_position": spaces.Box(-10, 0, shape=(2,), dtype=float),
            }
        )
        self.action_space = spaces.Discrete(4)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the self to access various robot methods

        self.front_left_motor = self.getDevice("fl_wheel_joint")
        self.front_right_motor = self.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.getDevice("rr_wheel_joint")

        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))

        self.front_left_motor.setVelocity(2.0)
        self.front_right_motor.setVelocity(2.0)
        self.rear_left_motor.setVelocity(2.0)
        self.rear_right_motor.setVelocity(2.0)

        camera_rgb = self.getDevice("camera rgb")
        camera_depth = self.getDevice("camera depth")
        camera_rgb.enable(self.TIME_STEP)
        camera_depth.enable(self.TIME_STEP)
        lidar = self.getDevice("laser")
        lidar.enable(self.TIME_STEP)
        # lidar.enablePointCloud()

