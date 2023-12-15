import sys
import gymnasium as gym
from sb3_contrib import ARS, TRPO, TQC

from training import latest_model
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC, A2C
from training import latest_model

env = gym.make('BipedalWalker-v3', hardcore=True, render_mode="human")


def test_model(algorithm, path):
    model = algorithm.load(path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, warn=False)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def model_selection(path):
    algorithm = path.split("/")[1].split("_")[0]

    if algorithm == "ARS":
        test_model(ARS, path)
    if algorithm == "A2C":
        test_model(A2C, path)
    elif algorithm == "PPO":
        test_model(PPO, path)
    elif algorithm == "SAC":
        test_model(SAC, path)
    elif algorithm == "TRPO":
        test_model(TRPO, path)
    elif algorithm == "TQC":
        test_model(TQC, path)
    else:
        print("Invalid algorithm.")
        sys.exit(1)


if __name__ == "__main__":

    if sys.argv[1].split("_")[-1] != "env.zip":
        print("Invalid file.")
        sys.exit(1)

    try:
        model_selection("final_models/"+sys.argv[1])
    except FileNotFoundError as e:
        print(f"No such file: '{sys.argv[1]}'")

