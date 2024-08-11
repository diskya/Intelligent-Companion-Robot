from robot_supervisor_env import RobotSupervisorEnv
from gymnasium import spaces
import numpy as np
from controller import Supervisor
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HomeNavigationSupervisor(RobotSupervisorEnv):

    def __init__(self):

        super().__init__()
        self.max_episode_steps = 200
        # Set up gym spaces
        self.observation_space = spaces.Dict(
            {
                "robot_position": spaces.Box(-10, 0, shape=(2,), dtype=np.float64),
                "distance": spaces.Box(0, 8, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Discrete(4)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        # self.raw_orientation = np.array(self.robot.getOrientation())
        # self.orientation = np.arctan2(self.raw_orientation[3], self.raw_orientation[0])
        # print("Orientation: ", self.orientation)
        self.wheels = [None for _ in range(4)]
        self.setup_robot()        

    def get_observations(self):
        robot_position = np.array(self.robot.getPosition()[:2])
        target_position = np.array([-8.23, -6.49])
        distance = np.linalg.norm(robot_position - target_position)
        distance = np.array([distance], dtype=np.float64)
        obs = {"robot_position": robot_position, "distance": distance}
        return obs
    
    def get_default_observation(self):
        robot_position = np.array(self.robot.getPosition()[:2])
        target_position = np.array([-8.23, -6.49])
        distance = np.linalg.norm(robot_position - target_position)
        distance = np.array([distance], dtype=np.float64)
        obs = {"robot_position": robot_position, "distance": distance}
        return obs

    def get_reward(self, obs):
        # Define the reward as an inverse of the distance
        # self.reward -= obs['distance'][0]  # Adding 1 to avoid division by zero
        self.reward -= 1
            
        # if self.steps >= self.max_episode_steps:
        #     self.terminated = True
        #     print("Max Steps Reached, total reward: ", self.reward)
        # elif obs['distance'] <= 0.5:
        #     self.reward += 1000
        #     self.terminated = True
        #     print("Goal Reached!, total reward: ", self.reward)
        # elif obs['distance'] > 1.5:
        #     self.terminated = True
        #     print("Too Far!, total reward: ", self.reward)

        # print("Reward: ", self.reward)
        # print(distance)
        return self.reward

    def is_done(self, obs):
        return bool(self.terminated)



    def apply_action(self, action):
        # assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)
        for i in range(len(self.wheels)):
                self.wheels[i].setPosition(float('inf'))

        if action == 0:
            self.wheels[0].setVelocity(6.0)
            self.wheels[1].setVelocity(6.0)
            self.wheels[2].setVelocity(6.0)
            self.wheels[3].setVelocity(6.0)
        elif action == 1:
            self.wheels[0].setVelocity(-4.0)
            self.wheels[1].setVelocity(-4.0)
            self.wheels[2].setVelocity(-4.0)
            self.wheels[3].setVelocity(-4.0)
        elif action == 2:
            self.wheels[0].setVelocity(-4.0)
            self.wheels[1].setVelocity(4.0)
            self.wheels[2].setVelocity(-4.0)
            self.wheels[3].setVelocity(4.0)
        elif action == 3:
            self.wheels[0].setVelocity(4.0)
            self.wheels[1].setVelocity(-4.0)
            self.wheels[2].setVelocity(4.0)
            self.wheels[3].setVelocity(-4.0)
        
        recognized_object_array = self.camera_rgb.getRecognitionObjects()
        array_len = len(recognized_object_array)
        if array_len == 0:
            print('searching for goal...')
        else: 
            print('goal found!')
            recognized_object = self.camera_rgb.getRecognitionObjects()[0]
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
