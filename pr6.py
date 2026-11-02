# Reinforcement Learning
# A. Implement Reinforcement Learning using an example of a maze environment that the 
# agent needs to explore. 


import numpy as np
import random

# Maze: 0 = Path, 1 = Wall
maze = [
    [0, 0, 0, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
]

start = (0, 0)
goal = (3, 3)

# Up, Down, Left, Right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Q-table
Q = np.zeros((4, 4, 4))

# Training
for episode in range(1000):

    state = start

    for step in range(100):

        r, c = state

        # Choose action
        if random.random() < 0.2:
            a = random.randint(0, 3)
        else:
            a = np.argmax(Q[r, c])

        # Move agent
        dr, dc = actions[a]
        nr, nc = r + dr, c + dc

        # Check movement
        if (nr < 0 or nr >= 4 or nc < 0 or nc >= 4
                or maze[nr][nc] == 1):
            next_state = state
            reward = -10
        else:
            next_state = (nr, nc)
            reward = 100 if next_state == goal else -1

        # Update Q-table
        Q[r, c, a] += 0.1 * (
            reward + 0.9 * np.max(Q[next_state]) - Q[r, c, a]
        )

        state = next_state

        if state == goal:
            break

# Display learned path
state = start
path = [state]

for step in range(20):

    r, c = state
    a = np.argmax(Q[r, c])

    dr, dc = actions[a]
    next_state = (r + dr, c + dc)

    if (next_state[0] < 0 or next_state[0] >= 4
            or next_state[1] < 0 or next_state[1] >= 4
            or maze[next_state[0]][next_state[1]] == 1):
        break

    state = next_state
    path.append(state)

    if state == goal:
        break

print("Learned Path:", path)

if state == goal:
    print("Goal Reached!")
else:
    print("Goal Not Reached!")
