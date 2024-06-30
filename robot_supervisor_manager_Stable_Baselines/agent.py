from cartpole_env import CartPoleRobotSupervisor
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import  check_env

def run():
    # Verify that the environment is working as a gym-style env
    #check_env(env)

    #  Use the PPO algorithm from the stable baselines having MLP, verbose=1  output the training information
    model = PPO("MlpPolicy", env, verbose=1)
    # Indicate the total timmepstes that the agent should be trained.
    model.learn(total_timesteps=40000)
    # Save the model
    model.save("ppo_cartpole")

    # End of the training period and now we evaluate the trained agent
    print("Training finished")
    evaluate()


def evaluate():
    print("Evaluating the trained agent")
    model = PPO.load("ppo_cartpole")
    obs = env.reset()
    env.episode_score = 0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.episode_score += reward  # Accumulate episode reward

        if done:
            print("Reward accumulated =", env.episode_score)
            env.episode_score = 0
            obs = env.reset()


if __name__ == "__main__":
    env = CartPoleRobotSupervisor()
    run()