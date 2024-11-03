import numpy as np


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        self.Q = np.zeros((self.n_states, self.n_actions))

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            max_value = np.max(self.Q[state])
            max_actions = np.where(self.Q[state] == max_value)[0]
            return np.random.choice(max_actions)

    def update_Q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self, n_episodes):
        steps_per_episode = []
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False

            steps = 0
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.update_Q(state, action, reward, next_state)
                state = next_state
                steps += 1

            steps_per_episode.append(steps)

            if episode % 100 == 0:
                print(f"Episode {episode}")

        return steps_per_episode
