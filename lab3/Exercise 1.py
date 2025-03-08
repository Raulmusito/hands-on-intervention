# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
 
# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)                              # displacement along Z-axis
q = np.array([0.2,0.5,0.4]).reshape(3,1)     # rotation around Z-axis (theta)
alpha = np.zeros(3)                          # displacement along X-axis
a = np.array([0.75, 0.5, 0.4])               # rotation around X-axis 
revolute = [True,True,True]                  # flags specifying the type of joints

# Setting desired position of end-effector to the current one
T = kinematics(d, q.flatten(), a, alpha) # flatten() needed if q defined as column vector !
sigma_d = T[-1][0:2,3].reshape(2,1)

# Simulation params
dt = 1.0/60.0
Tt = 60 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []
qvec1 = []
qvec2 = []
qvec3 = []
# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma_d
    global PPx, PPy
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)[:2,:]
    
    
    # Update control
    sigma = T[-1][:2, 3].reshape(2,1)                      # Current position of the end-effector

    err =  sigma_d - sigma                    # Error in position
    Jbar = J                                  # Task Jacobian
    pin_J = np.linalg.pinv(J)                 # Pseudo-inverse of the Jacobian
    P =  np.eye(3) - pin_J@ J         # Null space projector
    y = np.array([0*np.sin(t), 1*np.cos(t), 0*np.tan(t)]).reshape(3,1)       # Arbitrary joint velocity
    dq = np.linalg.pinv(Jbar)@ err + P @ y # Control signal
    q = q + dt * dq # Simulation update
    qvec1.append(q[0][0]) #append(q) # Save joint values for plotting
    qvec2.append(q[1][0]) #append(q) # Save joint values for plotting
    qvec3.append(q[2][0]) #append(q) # Save joint values for plotting


    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 60, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()

# Plot error distances
plt.figure()
plt.plot(tt[:len(qvec1)], qvec1, label="Q1")
plt.plot(tt[:len(qvec2)], qvec2, label="Q2")
plt.plot(tt[:len(qvec3)], qvec3, label="Q3")
plt.title('Joint Values Over Time')
plt.xlabel('Time [s]')
plt.ylabel('Joint Values [rads]')

plt.legend()
plt.show()