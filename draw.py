import numpy as np 
from numpy.random import randint
import matplotlib.pyplot as plt 

L = [1.09, 3.23, 4.70, 3.38, 4.55, 3.72, 4.64, 4.58, 3.83, 4.40, 5.2, 5.4, 5.6, 4.70, 4.81, 5.6, 5.2]
x = []
for i in range(len(L)):
	x.append((i+1)*300)
plt.plot(x, L)
plt.xlabel('Samples')
plt.ylabel('Reward')
plt.savefig('500_300', dpi = 800)

import gym
for i in range(10):
	print(np.random.rand())

'''
env = gym.make('Breakout-v0')
#print(env.action_space.n)
#print(env.unwrapped.get_action_meanings())
def test_run():
	score = 0
	done = 0
	env.reset()
	env.step(1)
	while not done:
		#env.render()
		_, reward, done, info = env.step(randint(0, env.action_space.n))
		#print(reward, done, info)
		score += reward
	env.close()
	print(score)
	return score
test_run()
a = []
for _ in range(100):
	a.append(test_run())
print(a)
print(np.mean(a))
'''