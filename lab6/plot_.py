import numpy as np
import matplotlib.pyplot as plt

simb = "base_history"
simee = "ee_history"

# Load 6 .npy files containing xy coordinates
vector1 = np.load(simb+'1.npy')
vector2 = np.load(simb+'2.npy')
vector3 = np.load(simb+'3.npy')
vector4 = np.load(simee+'1.npy')
vector5 = np.load(simee+'2.npy')
vector6 = np.load(simee+'3.npy')

# Plot all vectors on the same graph
plt.figure()
plt.plot(vector1[:, 0], vector1[:, 1], label='Base 1',  color="blue", linestyle='--')
plt.plot(vector2[:, 0], vector2[:, 1], label='Base 2', color="green", linestyle='--')
plt.plot(vector3[:, 0], vector3[:, 1], label='Base 3',   color="red", linestyle='--')
plt.plot(vector4[:, 0], vector4[:, 1], label='EE 1',    color="blue", linestyle='--')
plt.plot(vector5[:, 0], vector5[:, 1], label='EE 2',  color ="green", linestyle='--')
plt.plot(vector6[:, 0], vector6[:, 1], label='EE 3',     color="red", linestyle='--')

# Add labels, legend, and grid
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('XY Coordinates from base and ee for 3 different update methods')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()