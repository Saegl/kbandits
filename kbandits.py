import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def update(self, action, reward):
        pass

    @abstractmethod
    def reset(self):
        pass


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, k, epsilon, initial_q_value=0.0):
        self.k = k
        self.epsilon = epsilon
        self.initial_q_value = initial_q_value
        self.q_hat = np.full(k, initial_q_value)
        self.n = np.zeros(k)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.q_hat))
        else:
            return np.argmax(self.q_hat)

    def update(self, action, reward):
        self.n[action] += 1
        self.q_hat[action] += (1 / self.n[action]) * (reward - self.q_hat[action])

    def reset(self):
        self.q_hat = np.full(self.k, self.initial_q_value)
        self.n = np.zeros(self.k)


class UCBStrategy(Strategy):
    def __init__(self, k, initial_q_value=0.0, c=2):
        self.k = k
        self.initial_q_value = initial_q_value
        self.q_hat = np.full(k, initial_q_value)
        self.n = np.zeros(k)
        self.c = c
        self.t = 0

    def select_action(self):
        if 0 in self.n:
            return np.argmin(self.n)
        
        ucb_values = self.q_hat + self.c * np.sqrt(np.log(self.t) / self.n)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.n[action] += 1
        self.q_hat[action] += (1 / self.n[action]) * (reward - self.q_hat[action])
        self.t += 1

    def reset(self):
        self.q_hat = np.full(self.k, self.initial_q_value)
        self.n = np.zeros(self.k)
        self.t = 0


class Bandits:
    def __init__(self, seed, k):
        self.gen = np.random.default_rng(seed)
        self.k = k
        self.q_star = self.gen.uniform(0.0, 10.0, k)

    def pull(self, i) -> float:
        return self.gen.normal(self.q_star[i], 1.0)

    def run(self, strategy, runs, timesteps):
        total_rewards = np.zeros(timesteps)
        optimal_action_counts = np.zeros(timesteps)
        optimal_action = np.argmax(self.q_star)

        for run in range(runs):
            strategy.reset()
            rewards = np.zeros(timesteps)
            action_counts = np.zeros(self.k)

            for t in range(timesteps):
                action = strategy.select_action()
                reward = self.pull(action)
                strategy.update(action, reward)

                rewards[t] = reward
                action_counts[action] += 1
                total_rewards[t] += reward
                if action == optimal_action:
                    optimal_action_counts[t] += 1

        average_rewards = total_rewards / runs
        percentage_optimal_actions = (optimal_action_counts / runs) * 100

        return average_rewards, percentage_optimal_actions


def run_simulation_and_plot(k, runs, timesteps, strategies):
    bandits = Bandits(seed=13, k=k)
    results = {}
    
    for strategy_name, strategy in strategies.items():
        avg_reward, pct_optimal_action = bandits.run(strategy, runs, timesteps)
        results[strategy_name] = {
            "avg_reward": avg_reward,
            "pct_optimal_action": pct_optimal_action
        }

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for strategy_name, data in results.items():
        plt.plot(data["avg_reward"], label=strategy_name)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time (Across Runs)')
    plt.legend()

    plt.subplot(1, 2, 2)
    for strategy_name, data in results.items():
        plt.plot(data["pct_optimal_action"], label=strategy_name)
    plt.xlabel('Timesteps')
    plt.ylabel('Percentage Optimal Action')
    plt.title('Percentage of Optimal Action Over Time (Across Runs)')
    plt.legend()

    plt.tight_layout()
    plt.show()


k = 10
runs = 500
timesteps = 1000
epsilon_values = [0, 0.1, 0.01]
strategies = {
    "UCB": UCBStrategy(k=k, c=2)
}

for epsilon in epsilon_values:
    strategies[f"Epsilon={epsilon}"] = EpsilonGreedyStrategy(k=k, epsilon=epsilon, initial_q_value=0.0)

run_simulation_and_plot(k, runs, timesteps, strategies)
