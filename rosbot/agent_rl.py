from home_navigation_env import HomeNavigationSupervisor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import NormalizeObservation
import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = HomeNavigationSupervisor()
# env = DummyVecEnv([lambda: HomeNavigationSupervisor()])  # The environment is wrapped in a DummyVecEnv to make it compatible with stable baselines
# env = VecNormalize(env, norm_obs=False, norm_reward=False)
# check_env(env)


def train():
    #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
    model = PPO("MultiInputPolicy", env, verbose=1, gamma=0.99, learning_rate=0.0003, clip_range=0.1) # or "MultiInputPolicy" for multiple inputs
    model.policy.to(device)
    # Indicate the total timmepstes that the agent should be trained.
    model.learn(total_timesteps=2048)
    # Save the model
    model.save("navigation")
    # env.save("vec_normalize.pkl")

    # End of the training period and now we evaluate the trained agent
    print("Training finished")
    evaluate()


def evaluate():
    print("Playing the trained agent")
    # eval_env = VecNormalize.load("vec_normalize.pkl", env)
    eval_env = env

    model = PPO.load("navigation")
    obs, info = eval_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = eval_env.step(action)

        if done:
            obs, info = eval_env.reset()


if __name__ == "__main__":
    train()