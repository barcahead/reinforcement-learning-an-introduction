import numpy as np
import seaborn as sns
from tqdm import tqdm 

class Bandit:

	def __init__(self, k_arm = 10, epsilon = 0.1, step_size = 0.1, sample_average = False):
		self.k = k_arm
		self.epsilon = epsilon
		self.step_size = step_size
		self.sample_average = sample_average

		self.indices = np.arange(self.k)

		self.q_true = np.zeros(self.k)
		self.q_estimation = np.zeros(self.k)
		self.action_count = np.zeros(self.k)

		self.average_reward = 0
		self.time = 0

	def act(self):
		if np.random.rand() < self.epsilon:
			return np.random.choice(self.indices)
		return np.argmax(self.q_estimation)

	def step(self, action):
		reward = np.random.randn() + self.q_true[action]
		self.time += 1
		self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
		self.action_count[action] += 1

		if self.sample_average:
			self.q_estimation[action] += 1.0 / self.action_count[action] * (reward - self.q_estimation[action])
		else:
			self.q_estimation[action] += 1.0 / self.step_size * (reward - self.q_estimation[action]) 
		return reward

	def drift(self):
		for i in range(self.k):
			self.q_true[i] += 0.01 * np.random.randn()

def simulate(time, bandits):
	

def figure():



if __name__ == "__main__":
	figure()