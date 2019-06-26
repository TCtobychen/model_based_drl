import numpy as np 
from numpy.random import randint

for i in range(10):
	try:
		print(1/i)
	except ZeroDivisionError:
		continue
	print('HAHA')

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