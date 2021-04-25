import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sb


def get_score(env, policy, gamma):
	reward_list = []
	for i in range(50):
		observation = env.reset()
		total_reward_temp = 0
		step_count = 0
		while True:
			observation, reward, done, _ = env.step(int(policy[observation]))
			total_reward_temp += (gamma ** step_count * reward)
			step_count += 1
			if done: break
		reward_list.append(total_reward_temp)
	reward_mean = np.mean(reward_list)

	return reward_mean

def get_policy(env, statevalue, gamma):
	policy = np.zeros(env.nS)
	for istate in range(env.nS):
		action_values = []
		for action in range(env.nA):
			val = 0
			for action_executed in range(len(env.P[istate][action])):
				probability, next_state, reward, _ = env.P[istate][action][action_executed]
				val += probability * (reward + gamma * statevalue[next_state])
			action_values.append(val)
		policy[istate] = np.argmax(np.asarray(action_values))

	return policy

def get_value(env, policy, gamma):
	value = np.zeros(env.nS)
	theta = 1e-8
	while True:
		delta = 0
		for istate in range(env.nS):
			action = policy[istate]
			val_temp = 0
			for action_executed in range(len(env.P[istate][action])):
				probability, next_state, reward, _ = env.P[istate][action][action_executed]
				val_temp += probability * (reward + gamma * value[next_state])
			delta = max(delta, abs(value[istate] - val_temp))
			value[istate] = val_temp

		if (delta <= theta): break
	
	return value

def policy_iteration(env, gamma):
	policy = np.random.choice(env.nA, size=(env.nS))  
	max_iters = 10000
	for i in range(max_iters):
		value = get_value(env, policy, gamma)
		new_policy = get_policy(env, value, gamma)
		if (np.all(policy == new_policy)):
			n_iter = i+1
			break
		policy = np.copy(new_policy)
    
	return policy, n_iter

def value_iteration(env, gamma):
	value = np.zeros(env.nS)  
	max_iters = 10000
	theta = 1e-8
	for i in range(max_iters):
		delta = 0
		
		for istate in range(env.nS):
			v_temp_action = []
			for iaction in range(env.nA):
				val_temp = 0
				for action_executed in range(len(env.P[istate][iaction])):
					probability, next_state, reward, _ = env.P[istate][iaction][action_executed]
					val_temp += probability * (reward + gamma * value[next_state])
				v_temp_action.append(val_temp)
			temp = max(v_temp_action)
			delta = max(delta, abs(value[istate] - temp))
			value[istate] = temp
		if (delta <= theta): 
			n_iter = i + 1
			break
	return value, n_iter

def plot_policy_map(title, policy, map_desc, color_map, direction_map, value):
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

	plt.figure(title+" Value Map", figsize=(4, 4))
	sb.heatmap(value.reshape(4,4),  cmap="YlGnBu", annot=True, cbar=False)
	plt.title(title+" Value Map")
	plt.savefig(title+" Value Map"+str('.png'))
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
	
	gamma_list=[0.1*i+0.05 for i in range(10)]
	time_list=[]
	num_iters=[]
	score_list=[]
	
	#policy iteration
	for gamma in gamma_list:
		st=time.time()
		policy_converged, n_iter = policy_iteration(env, gamma)
		end=time.time()
		score = get_score(env, policy_converged, gamma)
		value = get_value(env, policy_converged, gamma)
		score_list.append(score)
		num_iters.append(n_iter)
		time_list.append(end-st)
		
		plot_policy_map('Policy Iteration '+ 'Gamma: ' + str(round(gamma,2)), policy_converged.reshape(4,4), desc, colors_lake(), directions_lake(), value)

	plt.plot(gamma_list, time_list)
	plt.xlabel('Gamma')
	plt.title('Run Time - Policy Iteration')
	plt.ylabel('Run Time (s)')
	plt.grid()
	plt.show()

	plt.plot(gamma_list,score_list)
	plt.xlabel('Gamma')
	plt.ylabel('Expected Reward')
	plt.title('Reward - Policy Iteration')
	plt.grid()
	plt.show()

	plt.plot(gamma_list,num_iters)
	plt.xlabel('Gamma')
	plt.ylabel('Iterations')
	plt.title('Convergence - Policy Iteration')
	plt.grid()
	plt.show()


FrozenLake()





