# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition
d = np.zeros(2)           
a = np.array([0.75, 0.5]) 
alpha = np.zeros(2)       
revolute = [True, True]
sigma_d = np.array([0.0, 1.0])
K = np.diag([2, 2, 0, 0, 0, 0])
K_DLS = np.diag([50, 50, 0, 0, 0, 0])

# Simulation params
dt = 1.0 / 60.0
tt = np.arange(0, 1000, dt)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Titles for subplots
titles = ["Transpose", "DLS", "Pseudo-Inverse"]

# Separate states for each method
q_transpose = np.array([0.2, 0.5])
q_DLS = np.array([0.2, 0.5])
q_Pinv = np.array([0.2, 0.5])

PPx_transpose, PPy_transpose = [], []
PPx_DLS, PPy_DLS = [], []
PPx_Pinv, PPy_Pinv = [], []

err_distance_transpose = []
err_distance_DLS = []
err_distance_Pinv = []

# Initialize plots for each simulation
lines, paths, points = [], [], []
for i in range(3):
    ax = axes[i]
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title(titles[i])
    ax.set_aspect('equal')
    ax.grid()
    
    line, = ax.plot([], [], 'o-', lw=2)  
    path, = ax.plot([], [], 'c-', lw=1)  
    point, = ax.plot([], [], 'rx')       
    
    lines.append(line)
    paths.append(path)
    points.append(point)

# Simulation initialization
def init():
    for i in range(3):
        lines[i].set_data([], [])
        paths[i].set_data([], [])
        points[i].set_data([], [])
    return lines + paths + points

# --- Simulation Functions ---
def simulate_transpose(t):
    global q_transpose

    # Update robot kinematics
    T = kinematics(d, q_transpose, a, alpha)
    J = jacobian(T, revolute)

    # Compute control error
    sigma = T[-1][:2, 3]       
    err = sigma_d - sigma      
    err = np.array([err[0], err[1], 0, 0, 0, 0])
    
    err_distance_transpose.append(err[0]**2 + err[1]**2)

    # Compute joint velocity
    dq = J.T @ (err @ K)
    q_transpose += dt * dq  

    # Update drawing
    P = robotPoints2D(T)
    PPx_transpose.append(P[0, -1])
    PPy_transpose.append(P[1, -1])
    lines[0].set_data(P[0, :], P[1, :])
    paths[0].set_data(PPx_transpose, PPy_transpose)
    points[0].set_data(sigma_d[0], sigma_d[1])

    return lines[0], paths[0], points[0]

def simulate_DLS(t):
    global q_DLS

    # Update robot kinematics
    T = kinematics(d, q_DLS, a, alpha)
    J = jacobian(T, revolute)

    # Compute control error
    sigma = T[-1][:2, 3]       
    err = sigma_d - sigma      
    err = np.array([err[0], err[1], 0, 0, 0, 0])
    
    err_distance_DLS.append(err[0]**2 + err[1]**2)

    # Compute joint velocity using DLS
    dq = DLS(J, 4) @ (err @ K_DLS)
    q_DLS += dt * dq  

    # Update drawing
    P = robotPoints2D(T)
    PPx_DLS.append(P[0, -1])
    PPy_DLS.append(P[1, -1])
    lines[1].set_data(P[0, :], P[1, :])
    paths[1].set_data(PPx_DLS, PPy_DLS)
    points[1].set_data(sigma_d[0], sigma_d[1])

    return lines[1], paths[1], points[1]

def simulate_Pinv(t):
    global q_Pinv

    # Update robot kinematics
    T = kinematics(d, q_Pinv, a, alpha)
    J = jacobian(T, revolute)

    # Compute control error
    sigma = T[-1][:2, 3]       
    err = sigma_d - sigma      
    err = np.array([err[0], err[1], 0, 0, 0, 0])
    
    err_distance_Pinv.append(err[0]**2 + err[1]**2)

    # Compute joint velocity using Pseudo-Inverse
    dq = np.linalg.pinv(J) @ (err @ K)
    q_Pinv += dt * dq  

    # Update drawing
    P = robotPoints2D(T)
    PPx_Pinv.append(P[0, -1])
    PPy_Pinv.append(P[1, -1])
    lines[2].set_data(P[0, :], P[1, :])
    paths[2].set_data(PPx_Pinv, PPy_Pinv)
    points[2].set_data(sigma_d[0], sigma_d[1])

    return lines[2], paths[2], points[2]

# Create animations using separate functions and states
animation1 = anim.FuncAnimation(fig, simulate_transpose, tt, 
                                interval=10, blit=True, init_func=init, repeat=True)
animation2 = anim.FuncAnimation(fig, simulate_DLS, tt, 
                                interval=10, blit=True, init_func=init, repeat=True)
animation3 = anim.FuncAnimation(fig, simulate_Pinv, tt, 
                                interval=10, blit=True, init_func=init, repeat=True)

plt.tight_layout()
plt.show()

# Plot error distances
plt.figure()
plt.plot(tt[:len(err_distance_transpose)], err_distance_transpose, label="Transpose")
plt.plot(tt[:len(err_distance_DLS)], err_distance_DLS, label="DLS")
plt.plot(tt[:len(err_distance_Pinv)], err_distance_Pinv, label="Pseudo-Inverse")
plt.title('Error Distance Over Time')
plt.xlabel('Time')
plt.ylabel('Error [m]')
plt.legend()
plt.show()
