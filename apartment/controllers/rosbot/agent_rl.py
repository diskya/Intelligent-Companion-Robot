from home_navigation_env import HomeNavigationSupervisor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from controller import Supervisor
import torch
import argparse

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse the arguments
parser = argparse.ArgumentParser(description="Agent RL Script")
parser.add_argument("-target", type=str, help="Specify the target task")
args = parser.parse_args()

# Create the environment
env = DummyVecEnv([lambda: HomeNavigationSupervisor(target=args.target)]) 
env = VecNormalize(env, norm_obs=True, norm_reward=False)


eval_env = env

model = PPO.load("navigation_BOB_2")
obs = eval_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = eval_env.step(action)

    if done:
    #     # break
        wheels = [None for _ in range(4)]
        wheels[0] = Supervisor.getDevice('fl_wheel_joint')
        wheels[1] = Supervisor.getDevice('fr_wheel_joint')
        wheels[2] = Supervisor.getDevice('rl_wheel_joint')
        wheels[3] = Supervisor.getDevice('rr_wheel_joint')
        for i in range(len(wheels)):
            wheels[i].setPosition(float('inf'))
            wheels[i].setVelocity(0.0)
        obs = eval_env.reset()
    #     print("Done")
    #     break