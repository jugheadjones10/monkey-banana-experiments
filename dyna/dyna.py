from collections import defaultdict

import numpy as np


class TabularDynaQ:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning_steps=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        actions_array = np.linspace(
            self.env.action_space.start,
            self.env.action_space.start + self.env.action_space.n - 1,
            num=self.env.action_space.n,
        )
        self.actions_array = actions_array.astype(int)

        # self.Q = np.zeros((self.n_states, self.n_actions))
        # Q should be dict keyed by states and actions
        self.Q = defaultdict(
            lambda: defaultdict(int, {i: 0 for i in self.actions_array})
        )
        self.model = {}

    def switch_env(self, env):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        actions_array = np.linspace(
            self.env.action_space.start,
            self.env.action_space.start + self.env.action_space.n - 1,
            num=self.env.action_space.n,
        )
        self.actions_array = actions_array.astype(int)

        self.Q = defaultdict(
            lambda: defaultdict(int, {i: 0 for i in self.actions_array})
        )

        # We do not reset the model because we want to keep past learnings.
        # self.model = {}

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            max_value = max(self.Q[state].values())
            max_actions = [
                key for key, value in self.Q[state].items() if value == max_value
            ]
            # Use max_actions as index on discrete action space and then make a random choice from there
            # actions_array = np.linspace(
            #     self.env.action_space.start,
            #     self.env.action_space.start + self.env.action_space.n - 1,
            #     num=self.env.action_space.n,
            # )
            # actions_array = actions_array.astype(int)

            return np.random.choice(max_actions)

    def update_Q(self, state, action, reward, next_state):
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def update_model(self, state, action, reward, next_state):
        self.model[(state, action)] = (reward, next_state)

    def clear_rewards_in_model(self):
        for key in self.model.keys():
            self.model[key] = (0, self.model[key][1])

    def plan(self):
        for _ in range(self.n_planning_steps):
            # Randomly sample previously observed state
            state = np.random.choice([s for s, a in self.model.keys()])
            actions = [a for s, a in self.model.keys() if s == state]
            action = np.random.choice(actions)
            reward, next_state = self.model[(state, action)]
            self.update_Q(state, action, reward, next_state)

    def train_for_final(self):
        steps_per_episode = []

        start_states = set()
        for monkey_x in range(1, self.env.size + 1):
            for chair_x in range(1, self.env.size + 1):
                for banana_x in range(1, self.env.size + 1):
                    # Concatenate the state into a string
                    start_states.add(
                        int(
                            str(monkey_x)
                            + str(1)
                            + str(chair_x)
                            + str(1)
                            + str(banana_x)
                            + str(2)
                        )
                    )

        for i, state in enumerate(start_states):
            print(i)
            state = self.env.index_to_state(state)
            state, _ = self.env.reset(start_state=state)
            done = False
            steps = 0
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.update_Q(state, action, reward, next_state)
                self.update_model(state, action, reward, next_state)

                # Only plan after goal is reached
                if done:
                    self.plan()

                state = next_state
                steps += 1

            steps_per_episode.append(steps)

        return steps_per_episode

    def train(self, n_episodes):
        steps_per_episode = []
        for episode in range(n_episodes):
            state, _ = self.env.reset(explore=True)
            done = False

            steps = 0
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # print(next_state)

                self.update_Q(state, action, reward, next_state)
                self.update_model(state, action, reward, next_state)
                self.plan()

                state = next_state
                steps += 1

            steps_per_episode.append(steps)

            # print("Episode", episode)
            # if episode % 100 == 0:
            #     print(f"Episode {episode}")

        return steps_per_episode
