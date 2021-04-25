import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example

def Forest_Experiments():
	state_num = 10
	r1 = 20
	r2 = 2
	p = 0.8
	gamma = 0.9
	value_f = []
	iters = []
	time_array = []
	gamma_arr = []

	P, R = mdptoolbox.example.forest(S=state_num, r1=r1, r2=r2, p=p)

	pi1 = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
	pi1.run()

	pi2 = mdptoolbox.mdp.ValueIteration(P, R, gamma)
	pi2.run()

	pi3 = mdptoolbox.mdp.QLearning(P, R, gamma,n_iter=100000)
	pi3.run()


	plt.plot(range(0,state_num), pi1.policy, 'bo')
	plt.plot(range(0,state_num), pi2.policy, 'ro')
	plt.plot(range(0,state_num), pi3.policy, 'k+')
	plt.xlabel('State')
	plt.ylabel('Policy')
	plt.legend(['PI','VI','QL'])
	plt.title('Optimal Policy'+' state_num='+str(state_num))
	plt.grid()
	plt.show()




	
	


	return

Forest_Experiments()





