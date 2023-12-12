from gymnasium import RewardWrapper as RW


class RewardWrapper(RW):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.penalty_limit = 500

    def reset(self, **kwargs):
        self.penalty_limit = 500
        return self.env.reset(**kwargs)

    def step(self, action):
        # Perform the environment step
        obs, reward, done, _, info = self.env.step(action)

        # print(f"Penalty limit: {self.penalty_limit}")
        if float(reward) < 0:
            self.penalty_limit -= 1
            if self.penalty_limit == 0:
                done = True
        else:
            self.penalty_limit = 500

        return obs, reward, done, _, info
