import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
import os

models_dir = "models/A2C"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('CarRacing-v2')
env.reset()

model = A2C(MlpPolicy, env, verbose=1)

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
