import sys
import gymnasium as gym
from training import latest_model
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC, A2C
from training import latest_model

env = gym.make('BipedalWalker-v3', hardcore=True, render_mode="human")


def is_valid_iteration(iteration, algorithm):
    try:
        latest_iter = int(latest_model(algorithm).split('/')[-1].split('.')[0])
        return 0 <= iteration <= latest_iter
    except ValueError:
        return False


def test_model(algorithm, algo_name, iteration):
    model = algorithm.load(f"models/{algo_name}/{iteration}.zip")
    mean_reward, std_reward = evaluate_policy(model, env,
                                              n_eval_episodes=1, warn=False)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def main(algorithm, iteration):
    if iteration == "latest":
        iteration = latest_model(algorithm).split('/')[-1].split('.')[0]
        if algorithm == "A2C":
            test_model(A2C, algorithm, iteration)
        elif algorithm == "PPO":
            test_model(PPO, algorithm, iteration)
        elif algorithm == "SAC":
            test_model(SAC, algorithm, iteration)
        else:
            print("Invalid algorithm.")
            sys.exit(1)

    elif iteration.isdigit() and is_valid_iteration(iteration, algorithm):
        if algorithm == "A2C":
            test_model(A2C, algorithm, iteration)
        elif algorithm == "PPO":
            test_model(PPO, algorithm, iteration)
        elif algorithm == "SAC":
            test_model(SAC, algorithm, iteration)
        else:
            print("Invalid algorithm.")
            sys.exit(1)

    else:
        print("Invalid algorithm or iteration.")
        sys.exit(1)


def test_untrained():
    model = PPO("MlpPolicy", env, verbose=1)
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            break


if __name__ == "__main__":

    if sys.argv[1] == "untrained":
        test_untrained()
        sys.exit(0)

    elif len(sys.argv) != 3:
        print("Usage: python script.py <algorithm> <iteration to test>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
