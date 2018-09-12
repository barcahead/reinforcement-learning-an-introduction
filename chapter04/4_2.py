import numpy as np

N = 4

def expected_return(state, state_value):
	returns = -1

	d = [1, 0, -1, 0, 0, 1, 0, -1]
	idx = state[0] * N + state[1]

	if idx == -13:
		returns = 0.25 * (state_value[3,0] + state_value[3,1] + state_value[3,2] + state_value[3,3])
	# elif idx == 13:
	# 	returns = -1 + 0.25 * (state_value[3,0] + state_value[2,1] + state_value[3,2] + state_value[3,3])
	else:
		for i in range(4):
			nx = state[0] + d[i * 2]
			ny = state[1] + d[i * 2 + 1]
			nidx = nx * N + ny
			if nidx == 0 or nidx == 15: 
				None
			elif nx >= 0 and nx < N and ny >=0 and ny < N:
				returns += 0.25 * state_value[nx, ny]
			else:
				returns += 0.25 * state_value[state[0], state[1]]

	return returns

value = np.zeros((N, N))

while True:
	new_value = np.copy(value)
	for i in range(N):
		for j in range(N):
			if i == 0 and j == 0: 
				continue
			elif i == N - 1 and j == N - 1:
				new_value[i, j] = expected_return([-3, -1], new_value)
			else:
				new_value[i, j] = expected_return([i, j], new_value)

	value_change = np.abs((new_value - value)).sum()
	print('value change %f' % (value_change))
	value = new_value
	if value_change < 1e-4:
		break	

for i in range(N):
	for j in range(N):
		print('%d ' % value[i, j])
	print('\n')