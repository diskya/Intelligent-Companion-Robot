import numpy as np

robot_position = np.array([-8.20, -6.20])
target_position = np.array([-8.25, -6.3])
distance = np.linalg.norm(robot_position - target_position)
distance = np.array([distance], dtype=float)

reward = distance[0]
print(distance)