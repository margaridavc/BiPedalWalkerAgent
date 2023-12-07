from gymnasium import RewardWrapper as RW


class RewardWrapper(RW):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.inside_track = True
        self.timout = 0

    def is_inside_track(self, obs):
        print(obs)
        """
        Check if the player is inside the track.

        Parameters:
        - obs: NumPy array representing the observation.

        Returns:
        - True if the player is inside the track, False otherwise.
        """
        # car_position =    TODO: Extract the car position
        # track =           TODO: Extract the track

        # Your existing logic to check if the player is within the track boundaries
        player_x, player_y = car_position
        track_height, track_width = track.shape[:2]

        # Check if the player is within the track boundaries
        if 0 <= player_x < track_width and 0 <= player_y < track_height:
            # Check if the track value at the player's position is within a valid range
            if 0 <= track[int(player_y), int(player_x)] <= 255:
                return True

        return False

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)

        self.inside_track = self.is_inside_track(obs)

        if not self.inside_track:
            reward -= 100
            self.timout += 1
            if self.timout == 10:
                done = True

        return obs, reward, done, _, info
