import sys
import gymnasium as gym
from training import latest_model
from stable_baselines3 import PPO
from stable_baselines3 import A2C


def is_valid_iteration(iteration, algorithm):
    try:
        latest_iter = int(latest_model(algorithm).split('/')[-1].split('.')[0])
        return 0 <= iteration <= latest_iter
    except ValueError:
        return False


def test_model(algorithm, algo_name, iteration):
    model = algorithm.load(f"models/{algo_name}/{iteration}0000")
    env = gym.make('CarRacing-v2', render_mode="human")
    obs, info = env.reset()
    done = False
    while done:
        action, _states = model.predict(obs)
        obs, rewards, done, _, info = env.step(action)
        env.render()
        print(rewards)


def main(algorithm, iteration):
    if not iteration.isdigit():
        print("Invalid iteration.")
        sys.exit(1)
    iteration = int(iteration)

    if is_valid_iteration(iteration, algorithm):
        if algorithm == "A2C":
            test_model(A2C, algorithm, iteration)
        elif algorithm == "PPO":
            test_model(PPO, algorithm, iteration)
        else:
            print("Invalid algorithm.")
            sys.exit(1)

    else:
        print("Invalid algorithm or iteration.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <algorithm> <iteration to test>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
