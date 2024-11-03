import numpy as np  # noqa
import pygame  # noqa
import logging

import gymnasium as gym  # noqa
from gymnasium import spaces  # noqa
from gymnasium.envs.registration import register  # noqa

# 1. Get reward when monkey reaches the x location of the banana
# 2. Get reward when monkey reaches the x location of the banana with the chair
# 3. Get reward when monkey reaches the y location of the banana after climbing the chair
# 4. Get reward when monkey climbs down from chair to reach the banana?


# We use 1-index instead of 0-index so that we can encode the state as concatenation of integers
class MonkeyBananaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The length of the line
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's, chair's, and banana's locations.
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([1, 1]), high=np.array([size + 1, 2]), dtype=int
                ),
                "chair": spaces.Box(
                    low=np.array([1, 1]), high=np.array([size + 1, 2]), dtype=int
                ),
                "banana": spaces.Box(
                    low=np.array([1, 1]), high=np.array([size + 1, 2]), dtype=int
                ),
            }
        )

        self.observation_space.n = int("525252")
        self.action_space = spaces.Discrete(6)
        self.visited_states = set()

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = {
            0: np.array([-1, 0]),  # "left"
            1: np.array([1, 0]),  # "right"
            2: np.array([-1, 0]),  # "move left with chair"
            3: np.array([1, 0]),  # "move right with chair"
            4: np.array([0, 1]),  # "climb"
            5: np.array([0, -1]),  # "climb down"
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        # So that we can display curr action in the pygame interface
        self.curr_action = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_obs(self):
        obs = {
            "agent": self._agent_location,
            "chair": self._chair_location,
            "banana": self._banana_location,
        }
        return self.state_to_index(obs)

    def state_to_index(self, state):
        return (
            state["agent"][0] * 100000
            + state["agent"][1] * 10000
            + state["chair"][0] * 1000
            + state["chair"][1] * 100
            + state["banana"][0] * 10
            + state["banana"][1]
        )

    def index_to_state(self, index):
        # Just separate the concatenated numbers to get back the state
        return {
            "agent": [int(str(index)[0]), int(str(index)[1])],
            "chair": [int(str(index)[2]), int(str(index)[3])],
            "banana": [int(str(index)[4]), int(str(index)[5])],
        }

    def _get_info(self):
        return ""

    def initialise_positions(self):
        # For agent, chair and banana
        self._agent_location = np.array(
            [
                self.np_random.integers(1, self.size + 1, size=1, dtype=int)[0],
                1,
            ]
        )

        self._chair_location = np.array(
            [
                self.np_random.integers(1, self.size + 1, size=1, dtype=int)[0],
                1,
            ]
        )

        self._banana_location = np.array(
            [
                self.np_random.integers(1, self.size + 1, size=1, dtype=int)[0],
                2,
            ]
        )

    # Make a reset function that purposefully sets the initial state to states that have not been visited
    def explore_reset(self):
        # We need the following line to seed self.np_random
        new_state = False

        max_iter = 50
        while not new_state and max_iter > 0:
            self.initialise_positions()
            observation = self._get_obs()
            if observation not in self.visited_states:
                new_state = True
            max_iter -= 1

        return observation

    def reset(self, seed=None, options=None, explore=False, start_state=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.initialise_positions()

        if start_state is not None:
            self._agent_location = start_state["agent"]
            self._chair_location = start_state["chair"]
            self._banana_location = start_state["banana"]

        observation = self._get_obs()

        if explore:
            observation = self.explore_reset()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def process_action(self, direction, action):
        # Move the agent or the agent with the chair
        if action in [0, 1]:  # Move left or right
            if self._agent_location[1] == 1:
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )

        elif action in [2, 3]:  # Move left or right with the chair
            if np.array_equal(self._agent_location, self._chair_location):
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )
                self._chair_location = np.clip(
                    self._chair_location + direction, [1, 1], [self.size, 2]
                )
        elif action == 4:  # Climb up the chair
            if (
                np.array_equal(self._agent_location, self._chair_location)
                and self._agent_location[1] == 1
            ):
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )
        elif action == 5:  # Climb down the chair
            if self._agent_location[1] == 2:
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )

    def step(self, action):
        self.curr_action = action

        terminated = False
        reward = -1

        # Determine the direction from the action
        direction = self._action_to_direction[action]

        self.process_action(direction, action)

        reward, terminated = self.goal_reached(reward, terminated)

        # Generate new observation and info
        observation = self._get_obs()
        self.visited_states.add(observation)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def goal_reached(self, reward, terminated):
        # Check if the agent has grabbed the banana
        # print(self._agent_location, self._banana_location)
        if np.array_equal(self._agent_location, self._banana_location):
            reward = 10  # Reward for grabbing the banana
            terminated = True  # End the episode

        return reward, terminated

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def action_index_to_label(self, action_index):
        if action_index == 0:
            return "left"
        elif action_index == 1:
            return "right"
        elif action_index == 2:
            return "left with chair"
        elif action_index == 3:
            return "right with chair"
        elif action_index == 4:
            return "climb"
        elif action_index == 5:
            return "climb down"
        else:
            return "unknown"

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Render the current action in the top left corner
        font = pygame.font.Font(None, 36)
        action_text = self.action_index_to_label(self.curr_action)
        text = font.render(action_text, True, (0, 0, 0))
        canvas.blit(text, (0, 0))

        # Function to flip y-coordinate
        def flip_y(coord):
            return np.array([coord[0], self.size - 1 - coord[1]])

        # Draw the chair
        pygame.draw.rect(
            canvas,
            (139, 69, 19),  # RGB for brown
            pygame.Rect(
                flip_y(self._chair_location - np.array([1, 1])) * pix_square_size,
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw the banana
        banana_image = pygame.image.load(
            "banana.png"
        )  # Ensure this image exists in your project directory
        banana_image = pygame.transform.scale(
            banana_image, (int(pix_square_size), int(pix_square_size))
        )
        banana_rect = banana_image.get_rect()
        banana_rect.center = (
            flip_y(self._banana_location - np.array([1, 1])) + 0.5
        ) * pix_square_size
        canvas.blit(banana_image, banana_rect)

        # Draw the agent (monkey)
        monkey_image = pygame.image.load(
            "monkey.jpeg"
        )  # Ensure this image exists in your project directory
        monkey_image = pygame.transform.scale(
            monkey_image, (int(pix_square_size), int(pix_square_size))
        )
        monkey_rect = monkey_image.get_rect()
        monkey_rect.center = (
            flip_y(self._agent_location - np.array([1, 1])) + 0.5
        ) * pix_square_size
        canvas.blit(monkey_image, monkey_rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class BananaOnFloorEnv(MonkeyBananaEnv):
    def __init__(self, render_mode=None, size=5):
        super().__init__(render_mode, size)
        self.action_space = spaces.Discrete(2)

    def goal_reached(self, reward, terminated):
        # Check if the agent has grabbed the banana
        if self._agent_location[0] == self._banana_location[0]:
            reward = 10  # Reward for grabbing the banana
            terminated = True  # End the episode

        return reward, terminated

    def process_action(self, direction, action):
        if action in [0, 1]:  # Move left or right
            if self._agent_location[1] == 1:
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )


class ReachBananaWithChairEnv(BananaOnFloorEnv):
    def __init__(self, render_mode=None, size=5):
        super().__init__(render_mode, size)
        self.action_space = spaces.Discrete(4)

    def goal_reached(self, reward, terminated):
        # Check if the agent has grabbed the banana
        if (
            self._agent_location[0] == self._banana_location[0]
            and self._chair_location[0] == self._banana_location[0]
            and np.array_equal(self._agent_location, self._chair_location)
        ):
            reward = 10  # Reward for grabbing the banana
            terminated = True  # End the episode

        return reward, terminated

    def process_action(self, direction, action):
        if action in [0, 1]:  # Move left or right
            if self._agent_location[1] == 1:
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )

        elif action in [2, 3]:  # Move left or right with the chair
            if np.array_equal(self._agent_location, self._chair_location):
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )
                self._chair_location = np.clip(
                    self._chair_location + direction, [1, 1], [self.size, 2]
                )


class ClimbToReachBananaEnv(MonkeyBananaEnv):
    def __init__(self, render_mode=None, size=5):
        super().__init__(render_mode, size)
        self.action_space = spaces.Discrete(1, start=4)

    # Override parent class method
    def initialise_positions(self):
        # Initialize positions
        common_x = self.np_random.integers(1, self.size + 1, size=1, dtype=int)[0]

        self._agent_location = np.array(
            [
                common_x,
                1,
            ]
        )

        self._chair_location = np.array(
            [
                common_x,
                1,
            ]
        )

        self._banana_location = np.array(
            [
                common_x,
                2,
            ]
        )

    def process_action(self, direction, action):
        if action == 4:  # Climb up the chair
            if (
                np.array_equal(self._agent_location, self._chair_location)
                and self._agent_location[1] == 1
            ):
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )


class ClimbDownEnv(MonkeyBananaEnv):
    def __init__(self, render_mode=None, size=5):
        super().__init__(render_mode, size)
        self.action_space = spaces.Discrete(1, start=5)

    def initialise_positions(self):
        super().initialise_positions()

        common_x = self.np_random.integers(1, self.size + 1, size=1, dtype=int)[0]

        self._agent_location = np.array(
            [
                common_x,
                2,
            ]
        )

        self._chair_location = np.array(
            [
                common_x,
                1,
            ]
        )

        self._banana_location = np.array(
            [
                common_x,
                2,
            ]
        )

    def goal_reached(self, reward, terminated):
        # Check if the agent has grabbed the banana
        if self._agent_location[1] == 1:
            reward = 10  # Reward for grabbing the banana
            terminated = True  # End the episode

        return reward, terminated

    def process_action(self, direction, action):
        if action == 5:  # Climb down the chair
            if self._agent_location[1] == 2:
                self._agent_location = np.clip(
                    self._agent_location + direction, [1, 1], [self.size, 2]
                )
