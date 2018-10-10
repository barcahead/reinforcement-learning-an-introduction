import numpy as np
from tqdm import tqdm

M1 = [
	'xxx.............f',
	'xx..............f',
	'xx..............f',
	'x...............f',
	'................f',
	'................f',
	'..........xxxxxxx',
	'.........xxxxxxxx',
	'.........xxxxxxxx',
	'.........xxxxxxxx',
	'.........xxxxxxxx',
	'.........xxxxxxxx',
	'.........xxxxxxxx',
	'.........xxxxxxxx',
	'x........xxxxxxxx',
	'x........xxxxxxxx',
	'x........xxxxxxxx',
	'x........xxxxxxxx',
	'x........xxxxxxxx',
	'x........xxxxxxxx',
	'x........xxxxxxxx',
	'xx.......xxxxxxxx',
	'xx.......xxxxxxxx',
	'xx.......xxxxxxxx',
	'xx.......xxxxxxxx',
	'xx.......xxxxxxxx',
	'xx.......xxxxxxxx',
	'xx.......xxxxxxxx',
	'xxx......xxxxxxxx',
	'xxx......xxxxxxxx',
	'xxxssssssxxxxxxxx'
	 ]

M2 = [
	'xxxxxxxxxxxxxxxx...............f',
	'xxxxxxxxxxxxx..................f',
	'xxxxxxxxxxxx...................f',
	'xxxxxxxxxxx....................f',
	'xxxxxxxxxxx....................f',
	'xxxxxxxxxxx....................f',
	'xxxxxxxxxxx....................f',
	'xxxxxxxxxxxx...................f',
	'xxxxxxxxxxxxx..................f',
	'xxxxxxxxxxxxxx................xx',
	'xxxxxxxxxxxxxx.............xxxxx',
	'xxxxxxxxxxxxxx............xxxxxx',
	'xxxxxxxxxxxxxx..........xxxxxxxx',
	'xxxxxxxxxxxxxx.........xxxxxxxxx',
	'xxxxxxxxxxxxx..........xxxxxxxxx',
	'xxxxxxxxxxxx...........xxxxxxxxx',
	'xxxxxxxxxxx............xxxxxxxxx',
	'xxxxxxxxxx.............xxxxxxxxx',
	'xxxxxxxxx..............xxxxxxxxx',
	'xxxxxxxx...............xxxxxxxxx',
	'xxxxxxx................xxxxxxxxx',
	'xxxxxx.................xxxxxxxxx',
	'xxxxx..................xxxxxxxxx',
	'xxxx...................xxxxxxxxx',
	'xxx....................xxxxxxxxx',
	'xx.....................xxxxxxxxx',
	'x......................xxxxxxxxx',
	'.......................xxxxxxxxx',
	'.......................xxxxxxxxx',
	'sssssssssssssssssssssssxxxxxxxxx'
	 ]

DISCOUNT = 0.9

EPSILON = 0.1

MAX_SPEED = 5

ACTIONS = [0, -1, 1]

def run(m, pi, starting_points, greedy = False):
	# trajectory (state, action, reward)
	width = len(m)
	height = len(m[0]) 

	starting_point = starting_points[np.random.choice(len(starting_points))]
	initial_state = (starting_point[0], starting_point[1], 0, 0)
	x, y, vx, vy = initial_state

	trajectory = []

	while True:
		# print(pi[x][y][vx][vy])
		if greedy:
			action_id = np.argmax(pi[x][y][vx][vy])
		else:
			action_id = np.random.choice(9, p = pi[x][y][vx][vy])
		# t = ((x, y, vx, vy), action_id)
		# print(t)
		trajectory.append(((x, y, vx, vy), action_id))

		vx += ACTIONS[action_id // 3]
		vy += ACTIONS[action_id % 3]
		next_x = x - vx
		next_y = y + vy
		if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height or m[next_x][next_y] == 'x':
			starting_point = starting_points[np.random.choice(len(starting_points))]
			initial_state = (starting_point[0], starting_point[1], 0, 0)
			x, y, vx, vy = initial_state
		elif m[next_x][next_y] == 'f':
			break
		else:
			x = next_x
			y = next_y

	return trajectory 


def monte_carlo_control_on_policy(m, episodes):
	# state (x, y, vx, vy)
	# action (ax, ay) 9
	# qa (state, action)
	# pi (state, action)
	width = len(m) 
	height = len(m[0])
	q = (np.random.rand(width, height, MAX_SPEED + 1, MAX_SPEED + 1, 9) + 1) * -100000000
	pi = np.zeros(q.shape)
	returns = np.zeros(q.shape)
	returns_count = np.zeros(q.shape)

	# an arbitrary epsilon soft policy
	for x in range(width):
		for y in range(height):
			for vx in range(MAX_SPEED + 1):
				for vy in range(MAX_SPEED + 1):
					valid_actions = []
					for a in range(9):
						next_vx = vx + ACTIONS[a // 3]
						next_vy = vy + ACTIONS[a % 3]
						if next_vx > MAX_SPEED or next_vy > MAX_SPEED or next_vx < 0 or next_vy < 0 or (next_vx == 0 and next_vy == 0):
							continue
						valid_actions.append(a)	

					action_num = len(valid_actions)

					max_action = np.random.choice(valid_actions)

					for a in valid_actions:
						if a == max_action:
							pi[x][y][vx][vy][a] = 1 - EPSILON + EPSILON / action_num
						else:
							pi[x][y][vx][vy][a] = EPSILON / action_num

	starting_points = []
	for i in range(height):
		if m[width-1][i] == 's':
			starting_points.append((width - 1, i))

	for i in tqdm(range(episodes)):
		trajectory = run(m, pi, starting_points)
		first_visit = {}
		for i, (state, _) in enumerate(trajectory):
			if state not in first_visit:
				first_visit[state] = i

		g = 0
		for i, (state, action) in reversed(list(enumerate(trajectory))):
			g  = DISCOUNT * g - 1
			if first_visit[state] != i:
				continue
			x, y, vx, vy = state
			returns_count[x][y][vx][vy][action] += 1
			returns[x][y][vx][vy][action] += (g - returns[x][y][vx][vy][action]) / returns_count[x][y][vx][vy][action]
			q[x][y][vx][vy][action]	= returns[x][y][vx][vy][action]

			max_action = np.argmax(q[x][y][vx][vy])

			if q[x][y][vx][vy][max_action] == 0:
				continue

			action_num = np.count_nonzero(pi[x][y][vx][vy])
			for i, a in enumerate(pi[x][y][vx][vy]):
				if i == max_action:
					pi[x][y][vx][vy][i] = 1 - EPSILON + EPSILON / action_num
				elif a > 0:
					pi[x][y][vx][vy][i] = EPSILON / action_num
			# print(pi[x][y][vx][vy])

		print(len(trajectory))

	# display one path from start to finish line
	trajectory = run(m, pi, starting_points, True)
	m_final = m.copy()
	for (x, y, _, _), _ in trajectory:
		m_final[x]= m_final[x][:y] + 'P' + m_final[x][y+1:]
	for x in range(width):
		print(m_final[x])


monte_carlo_control_on_policy(M1, 1000)
monte_carlo_control_on_policy(M2, 10000)


