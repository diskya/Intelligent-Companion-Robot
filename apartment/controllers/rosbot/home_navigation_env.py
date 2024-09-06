from robot_supervisor_env import RobotSupervisorEnv
from gymnasium import spaces
import numpy as np
from controller import Supervisor
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HomeNavigationSupervisor(RobotSupervisorEnv):

    def __init__(self, target):

        super().__init__()
        self.max_episode_steps = 100
        # Set up observation space
        self.observation_space = spaces.Box(low=np.array([0, -2*np.pi]),
                                            high=np.array([6, 2*np.pi]),
                                            dtype=np.float64)
        # Set up action space
        self.action_space = spaces.Discrete(3)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.target = self.getFromDef(target)  # Grab the target reference from the supervisor to access various target methods

        robot_position = np.array(self.robot.getPosition()[:2])
        target_position = np.array(self.target.getPosition()[:2])
        self.initial_distance = np.linalg.norm(robot_position - target_position)

        # Set up the robot
        self.wheels = [None for _ in range(4)]
        self.setup_robot()        

    def get_observations(self):
        # Get the distance between the robot and the target
        robot_position = np.array(self.robot.getPosition()[:2])
        target_position = np.array(self.target.getPosition()[:2])
        distance = np.linalg.norm(robot_position - target_position)

        # Get the orientation of the robot
        raw_orientation = self.robot.getOrientation()
        orientation = np.arctan2(raw_orientation[3], raw_orientation[0])
        vector_to_target = target_position - robot_position
        angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        orientation = angle_to_target - orientation
        orientation = (orientation + np.pi) % (2 * np.pi) - np.pi

        # Return the observations
        obs = np.array([distance, orientation], dtype=np.float64)
        return obs
    
    def get_default_observation(self):
        robot_position = np.array(self.robot.getPosition()[:2])
        target_position = np.array(self.target.getPosition()[:2])
        distance = np.linalg.norm(robot_position - target_position)
        self.initial_distance = np.linalg.norm(robot_position - target_position)

        # Get the orientation of the robot
        raw_orientation = self.robot.getOrientation()
        orientation = np.arctan2(raw_orientation[3], raw_orientation[0])
        vector_to_target = target_position - robot_position
        angle_to_target = np.arctan2(vector_to_target[1], vector_to_target[0])
        orientation = angle_to_target - orientation
        orientation = (orientation + np.pi) % (2 * np.pi) - np.pi



        # self.previous_distance = distance
        # self.previous_orientation = orientation
        # self.minimum_distance = distance
        # self.minimum_orientation = orientation
        obs = np.array([distance, orientation], dtype=np.float64)
        return obs

    def get_reward(self, obs):
        distance = obs[0]
        orientation = obs[1]
        # print("Distance: ", distance, "Orientation: ", orientation)
        if distance <= 0.8:
            self.reward += 1000
            self.terminated = True
            print("Goal Reached!, total reward: ", self.reward)
        elif self.steps >= self.max_episode_steps:
            self.truncated = True
            print("Max steps reached", self.reward)
            self.reset()

        self.reward -= 6
        normalized_distance = super().normalize_to_range(distance, 0, self.initial_distance, 0, 1.0)
        # print("orientation: ", orientation)
        if normalized_distance >= 0.8:
            self.reward += 0
        elif 0.6 < normalized_distance < 0.8:
            self.reward += 1
        elif 0.4 < normalized_distance <= 0.6:
            self.reward += 2
        elif 0.2 < normalized_distance <= 0.4:
            self.reward += 3
        elif normalized_distance <= 0.2:
            self.reward += 4

        if abs(orientation) < 0.5:
            self.reward += 2

        return self.reward

    def is_done(self, obs):
        return bool(self.terminated)



    def apply_action(self, action):
        # assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)
        for i in range(len(self.wheels)):
                self.wheels[i].setPosition(float('inf'))
                self.wheels[i].setVelocity(0.0)

        if action == 0:
            self.wheels[0].setVelocity(6.0)
            self.wheels[1].setVelocity(6.0)
            self.wheels[2].setVelocity(6.0)
            self.wheels[3].setVelocity(6.0)
        if action == 1:
            self.wheels[0].setVelocity(-4.0)
            self.wheels[1].setVelocity(4.0)
            self.wheels[2].setVelocity(-4.0)
            self.wheels[3].setVelocity(4.0)
        elif action == 2:
            self.wheels[0].setVelocity(4.0)
            self.wheels[1].setVelocity(-4.0)
            self.wheels[2].setVelocity(4.0)
            self.wheels[3].setVelocity(-4.0)

        super(Supervisor, self).step(self.timestep)

    def setup_robot(self):
        self.wheels[0] = self.getDevice('fl_wheel_joint')
        self.wheels[1] = self.getDevice('fr_wheel_joint')
        self.wheels[2] = self.getDevice('rl_wheel_joint')
        self.wheels[3] = self.getDevice('rr_wheel_joint')
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)
        
        self.camera_rgb = self.getDevice("camera rgb")
        # camera_depth = self.getDevice("camera depth")
        self.camera_rgb.enable(16)
        # camera_depth.enable(16)
        # self.camera_rgb.recognitionEnable(16)
        # self.camera_rgb.enableRecognitionSegmentation()
        # lidar = self.getDevice("laser")
        # lidar.enable(self.timestep)


    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}
