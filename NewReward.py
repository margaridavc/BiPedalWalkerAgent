from gymnasium import RewardWrapper as RW


class RewardWrapper(RW):
    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 25

    def reset(self, **kwargs):
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Perform the environment step
        obs, reward, done, _, info = self.env.step(action)

        self.total_reward += reward

        if self.total_reward < 0:
            done = True

        return obs, reward, done, _, info
