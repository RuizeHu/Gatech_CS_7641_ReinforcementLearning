import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sb
import random

def plot_policy_map(title, policy, map_desc, color_map, direction_map):
	fig = plt.figure()
	ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
	font_size = 'x-large'
	plt.title(title)
	for i in range(policy.shape[0]):
		for j in range(policy.shape[1]):
			y = policy.shape[0] - i - 1
			x = j
			p = plt.Rectangle([x, y], 1, 1)
			p.set_facecolor(color_map[map_desc[i,j]])
			ax.add_patch(p)

			ax.text(x+0.5, y+0.5, direction_map[policy[i, j]], weight='bold', size=font_size,
						   horizontalalignment='center', verticalalignment='center', color='w')
			
	plt.axis('off')
	plt.xlim((0, policy.shape[1]))
	plt.ylim((0, policy.shape[0]))
	plt.tight_layout()
	plt.savefig(title+str('.png'))
	plt.close()

	return (plt)

def colors_lake():
	return {
		b'S': 'green',
		b'F': 'blue',
		b'H': 'red',
		b'G': 'gold',
	}

def directions_lake():
	return {
		3: '⬆',
		2: '➡',
		1: '⬇',
		0: '⬅'
	}


# The main function
def FrozenLake():
	environment  = 'FrozenLake-v0'
	env = gym.make(environment)
	env = env.unwrapped
	desc = env.unwrapped.desc
	
	qtable = np.zeros((env.nS, env.nA))
	gamma_list=[0.05*(i+1) for i in range(20)]
	time_list=[]
	num_iters=[]
	score_list=[]
	gamma = 0.95

	total_episodes = 30000
	learning_rate = 0.1
	max_steps = 1000
	# exploration parameters
	epsilon = 1.0
	max_epsilon = 1.0
	min_epsilon = 0.01
	decay_rate = 0.001
	
	#Q learning
	rewards = []
	episodes = []
	bins = np.linspace(0, total_episodes, 101)
	total_reward_bined = []
	lb = -1
	ub = 0
	st=time.time()
	for episode in range(total_episodes):
		state = env.reset()
		done = False
		total_rewards = 0

		for step in range(max_steps):
			exp_exp_tradeoff = random.uniform(0,1)
            # determine if to explore or exploit
			if exp_exp_tradeoff > epsilon:
				action = np.argmax(qtable[state,:])
			else:
				action = env.action_space.sample()
			
            # take action and see outcome
			new_state, reward, done, _ = env.step(action)
			# update Q table
			qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
			# update total reward
			total_rewards += reward
			# update state
			state = new_state
			# end the game
			if done == True:
				break
		
		# vary rate of exploration
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
		rewards.append(total_rewards)
		episodes.append(episode)

		# count the number of episodes that achieved an award
		if episode % int(total_episodes/100) == 0:
			reward_temp = 0
			lb += 1
			ub += 1
			num_count = 0
		num_count += 1
		if episode >= bins[lb] and episode < bins[ub]:
			reward_temp += total_rewards
		if num_count == 100:
			total_reward_bined.append(reward_temp)

	policy = np.zeros(env.nS)
	for istate in range(env.nS):
		action = np.argmax(qtable[istate,:])
		policy[istate] = action
	end=time.time()
	print(end-st)

	plot_policy_map('Q Learning '+ 'Gamma: ' + str(round(gamma,2)), policy.reshape(4,4),desc,colors_lake(),directions_lake())


	plt.plot(np.linspace(100, total_episodes, 100), total_reward_bined)
	plt.xlabel('Episode')
	plt.title('Total Reward')
	plt.ylabel('Reward Per 100 Episodes')
	plt.grid()
	plt.show()


FrozenLake()





