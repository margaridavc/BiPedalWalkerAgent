from gymnasium import RewardWrapper as RW


class RewardWrapper(RW):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        # Perform the environment step
        obs, reward, done, _, info = self.env.step(action)

        # Add a reward for keeping balance
        # obs[2] is the angle of the agent from the vertical position
        balance_reward = abs(obs[2])
        reward += balance_reward

        return obs, reward, done, _, info
