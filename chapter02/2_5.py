import numpy as np
# import seaborn as sns
# import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm 

# sns.set(style = "darkgrid")

class Bandit:

	def __init__(self, k_arm = 10, epsilon = 0.1, step_size = 0.1, sample_average = False):
		self.k = k_arm
		self.epsilon = epsilon
		self.step_size = step_size
		self.sample_average = sample_average

		self.indices = np.arange(self.k)

	def reset(self):
		self.q_true = np.zeros(self.k)
		self.q_estimation = np.zeros(self.k)
		self.action_count = np.zeros(self.k)

	def act(self):
		if np.random.rand() < self.epsilon:
			return np.random.choice(self.indices)
		return np.argmax(self.q_estimation)

	def step(self, action):
		reward = np.random.randn() + self.q_true[action]
		self.action_count[action] += 1

		if self.sample_average:
			self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
		else:
			self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action]) 
		return reward

	def drift(self):
		for i in range(self.k):
			self.q_true[i] += 0.01 * np.random.randn()

def simulate(runs, time, bandits):
	best_action_counts = np.zeros((len(bandits), runs, time))
	rewards = np.zeros(best_action_counts.shape)
	for i, bandit in enumerate(bandits):
		for r in tqdm(range(runs)):
			bandit.reset()
			for t in range(time):
				action = bandit.act()
				reward = bandit.step(action)
				bandit.drift()
				rewards[i, r, t] = reward
	rewards = rewards.mean(axis = 1)
	return best_action_counts, rewards		

def figure(runs = 2000, time = 10000):
	bandits = [Bandit(sample_average = True), Bandit()]
	best_action_counts, rewards = simulate(runs, time, bandits)

	plt.plot(rewards[0], label = 'sample average')
	plt.plot(rewards[1], label = 'weigted average')
	plt.xlabel('steps')
	plt.ylabel('average reward')
	plt.legend()

	plt.savefig('../images/exercise_2_5.png')
	plt.close()

if __name__ == "__main__":
	figure()