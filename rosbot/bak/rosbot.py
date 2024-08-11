from controller import Supervisor
from controller import Robot, Keyboard
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env
from home_robot_env import HomeRobotEnv


supervisor = Supervisor() 
env = HomeRobotEnv()

while supervisor.step(env.timestep) != -1:
    pass


# def train():
#     #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
#     model = PPO("MultiInputPolicy", env, verbose=1) # or "MultiInputPolicy" for multiple inputs
#     # Indicate the total timmepstes that the agent should be trained.
#     model.learn(total_timesteps=40000)
#     # Save the model
#     model.save("home_robot")

#     # End of the training period and now we evaluate the trained agent
#     print("Training finished")
#     play()


# def play():
#     print("Playing the trained agent")
#     model = PPO.load("home_robot")
#     obs, info = env.reset()
#     env.episode_score = 0
#     while True:
#         action, _states = model.predict(obs)
#         obs, reward, done, truncated, _ = env.step(action)
#         env.episode_score += reward  # Accumulate episode reward

#         if done:
#             print("Reward accumulated =", env.episode_score)
#             env.episode_score = 0
#             obs = env.reset()


# if __name__ == "__main__":
#     train()


##########################################################################

# TIME_STEP = 128

# supervisor = Supervisor()  # create Supervisor instance

# front_left_motor = supervisor.getDevice("fl_wheel_joint")
# front_right_motor = supervisor.getDevice("fr_wheel_joint")
# rear_left_motor = supervisor.getDevice("rl_wheel_joint")
# rear_right_motor = supervisor.getDevice("rr_wheel_joint")

# front_left_motor.setPosition(float('inf'))
# front_right_motor.setPosition(float('inf'))
# rear_left_motor.setPosition(float('inf'))
# rear_right_motor.setPosition(float('inf'))

# front_left_motor.setVelocity(0.0)
# front_right_motor.setVelocity(0.0)
# rear_left_motor.setVelocity(0.0)
# rear_right_motor.setVelocity(0.0)

# camera_rgb = supervisor.getDevice("camera rgb")
# camera_depth = supervisor.getDevice("camera depth")
# camera_rgb.enable(TIME_STEP)
# camera_depth.enable(TIME_STEP)
# lidar = supervisor.getDevice("laser")
# lidar.enable(TIME_STEP)
# # lidar.enablePointCloud()

# ROSBOT = supervisor.getFromDef('ROSBOT')

# keyboard = Keyboard()
# keyboard.enable(TIME_STEP)

# while supervisor.step(TIME_STEP) != -1:
#     key = keyboard.getKey()
    
#     if key == Keyboard.UP:
#         front_left_motor.setVelocity(10)
#         front_right_motor.setVelocity(10)
#         rear_left_motor.setVelocity(10)
#         rear_right_motor.setVelocity(10)
#     elif key == Keyboard.DOWN:
#         front_left_motor.setVelocity(-10)
#         front_right_motor.setVelocity(-10)
#         rear_left_motor.setVelocity(-10)
#         rear_right_motor.setVelocity(-10)
#     elif key == Keyboard.LEFT:
#         front_left_motor.setVelocity(-4)
#         front_right_motor.setVelocity(4)
#         rear_left_motor.setVelocity(-4)
#         rear_right_motor.setVelocity(4)
#     elif key == Keyboard.RIGHT:
#         front_left_motor.setVelocity(4)
#         front_right_motor.setVelocity(-4)
#         rear_left_motor.setVelocity(4)
#         rear_right_motor.setVelocity(-4)
#     else:
#         front_left_motor.setVelocity(0)
#         front_right_motor.setVelocity(0)
#         rear_left_motor.setVelocity(0)
#         rear_right_motor.setVelocity(0)

#     print(ROSBOT.getPosition()[0])