import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example

def Forest_Experiments():
	state_num = 500
	r1 = 20
	r2 = 2
	p = 0.1
	gamma0 = 0.1
	value_f = []
	iters1 = []
	time_array1 = []
	val1 = []
	iters2 = []
	time_array2 = []
	val2 = []
	gamma_arr = []
	for i in range(1,11):
		gamma = gamma0 * i - 0.05
		P, R = mdptoolbox.example.forest(S=state_num, r1=r1, r2=r2, p=p)
		pi1 = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
		pi1.run()
		pi2 = mdptoolbox.mdp.ValueIteration(P, R, gamma)
		pi2.run()
		gamma_arr.append(gamma)
		iters1.append(pi1.iter)
		time_array1.append(pi1.time)
		val1.append(np.mean(pi1.V))
		iters2.append(pi2.iter)
		time_array2.append(pi2.time)
		val2.append(np.mean(pi2.V))


	plt.plot(gamma_arr, time_array1)
	plt.plot(gamma_arr, time_array2)
	plt.xlabel('Gamma')
	plt.title('Forest Management - Run time')
	plt.ylabel('Run Time (s)')
	plt.legend(['PI','VI'])
	plt.grid()
	plt.show()

	plt.plot(gamma_arr, iters1)
	plt.plot(gamma_arr, iters2)
	plt.xlabel('Gamma')
	plt.title('Forest Management - Convergence')
	plt.ylabel('Number of Iterations')
	plt.legend(['PI','VI'])
	plt.grid()
	plt.show()




	
	


	return

Forest_Experiments()





