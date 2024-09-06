from home_navigation_env import HomeNavigationSupervisor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from controller import Supervisor
import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# env = HomeNavigationSupervisor()
env = DummyVecEnv([lambda: HomeNavigationSupervisor(target='BOB')]) 
env = VecNormalize(env, norm_obs=True, norm_reward=False)
# check_env(env)


def train():
    #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.9, learning_rate=0.0003, clip_range=0.2) 
    # model = PPO.load("navigation")
    # model.set_env(env)
    model.policy.to(device)
    # Indicate the total timmepstes that the agent should be trained.
    model.learn(total_timesteps=100000)
    # Save the model
    model.save("navigation_BOB")
    # env.save("vec_normalize.pkl")

    # End of the training period and now we evaluate the trained agent
    print("Training finished")
    evaluate()

def continue_train():
    #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
    model = PPO.load("navigation_BOB")
    model.set_env(env)
    model.policy.to(device)
    # Indicate the total timmepstes that the agent should be trained.
    model.learn(total_timesteps=300000)
    # Save the model
    model.save("navigation_BOB_2")
    # env.save("vec_normalize.pkl")

    # End of the training period and now we evaluate the trained agent
    print("Training finished")
    evaluate()

def evaluate():
    print("Playing the trained agent")
    # eval_env = VecNormalize.load("vec_normalize.pkl", env)
    
    eval_env = env

    model = PPO.load("navigation_BOB_2")
    obs = eval_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)

        if done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            wheels = [None for _ in range(4)]
            wheels[0] = Supervisor.getDevice('fl_wheel_joint')
            wheels[1] = Supervisor.getDevice('fr_wheel_joint')
            wheels[2] = Supervisor.getDevice('rl_wheel_joint')
            wheels[3] = Supervisor.getDevice('rr_wheel_joint')
            for i in range(len(wheels)):
                wheels[i].setPosition(float('inf'))
                wheels[i].setVelocity(0.0)
            obs = eval_env.reset()


    # env = DummyVecEnv([lambda: HomeNavigationSupervisor(target='BOB')]) 
    # eval_env = VecNormalize(env, norm_obs=True, norm_reward=False)
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, _ = eval_env.step(action)

    #     if done:
    #         break


if __name__ == "__main__":
    evaluate()