import os
from sys import argv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as PPO_MlpPolicy
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy as A2C_MlpPolicy


env = gym.make('CarRacing-v2')
env.reset()
TIMESTEPS = 10000
models_dir = "models/"
logdir = "logs"


def latest_model(algorithm):
    models = [int(model.split(".")[0]) for model in os.listdir(f"models/{algorithm}")]
    models.sort()
    return f"models/{algorithm}/{models[-1]}.zip"


def train_A2C_model():
    if os.path.exists(f"{models_dir}/A2C"):
        if os.listdir(f"{models_dir}/A2C"):
            model_path = latest_model("A2C")
            a2c_model = A2C.load(model_path, env=env)
            iters = int(int(model_path.split("/")[2].split(".")[0]) / 10 ** 4)

        else:
            a2c_model = A2C(A2C_MlpPolicy, env, verbose=1, tensorboard_log=logdir)
            iters = 0
    else:
        os.makedirs(f"{models_dir}/A2C")
        a2c_model = A2C(A2C_MlpPolicy, env, verbose=1, tensorboard_log=logdir)
        iters = 0

    while True:
        iters += 1
        a2c_model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
        a2c_model.save(f"{models_dir}/A2C/{TIMESTEPS * iters}")


def train_PPO_model():
    if os.path.exists(f"{models_dir}/PPO"):
        if os.listdir(f"{models_dir}/PPO"):
            model_path = latest_model("PPO")
            ppo_model = PPO.load(model_path, env=env)
            iters = int(int(model_path.split("/")[2].split(".")[0]) / 10 ** 4)

        else:
            ppo_model = PPO(PPO_MlpPolicy, env, verbose=1, tensorboard_log=logdir)
            iters = 0
    else:
        os.makedirs(f"{models_dir}/PPO")
        ppo_model = PPO(PPO_MlpPolicy, env, verbose=1, tensorboard_log=logdir)
        iters = 0

    while True:
        iters += 1
        ppo_model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        ppo_model.save(f"{models_dir}/PPO/{TIMESTEPS * iters}")


if __name__ == '__main__':

    try:
        # Check if an argument is provided
        if len(argv) != 2:
            raise ValueError("No arguments given. Please specify which model to train.")

        model_type = argv[1]

        if model_type == "A2C":
            train_A2C_model()
        elif model_type == "PPO":
            train_PPO_model()
        else:
            raise ValueError("Invalid argument. Please specify 'A2C' or 'PPO'.")

    except ValueError as e:
        # Handling the specific exception (ValueError in this case)
        print(f"Error: {e}")
