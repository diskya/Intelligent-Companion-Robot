from robot_supervisor_env import RobotSupervisorEnv
from gym.spaces import Box, Discrete
import numpy as np


class CartPoleRobotSupervisor(RobotSupervisorEnv):

    def __init__(self):

        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        self.action_space = Discrete(2)

        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)

        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

        # Set up misc
        self.steps_per_episode = 200  # How many steps to run each episode (changing this messes up the solved condition)
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved

    def get_observations(self):
        # Position on x axis
        cart_position = super().normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x axis
        cart_velocity = super().normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Pole angle off vertical
        pole_angle = super().normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        endpoint_velocity = super().normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)

        return np.array([cart_position, cart_velocity, pole_angle, endpoint_velocity])

    def get_reward(self, action):
        return 1

    def is_done(self):
        # if self.episode_score > 195.0:
        #     return True

        pole_angle = round(self.position_sensor.getValue(), 2)
        if abs(pole_angle) > 0.261799388:  # 15 degrees off vertical
            return True

        cart_position = round(self.robot.getPosition()[0], 2)  # Position on x axis
        if abs(cart_position) > 0.39:
            return True

        return False

    def get_default_observation(self):
        return np.array([0.0 for _ in range(self.observation_space.shape[0])])

    def apply_action(self, action):
        assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(action)

        if action == 0:
            motor_speed = 5.0
        else:
            motor_speed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)

    def setup_motors(self):
        self.wheels[0] = self.getDevice('wheel1')
        self.wheels[1] = self.getDevice('wheel2')
        self.wheels[2] = self.getDevice('wheel3')
        self.wheels[3] = self.getDevice('wheel4')
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}
