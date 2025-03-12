from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

# Robot model - 3-link manipulator
d = np.zeros(3)                              # displacement along Z-axis
theta = np.array([0.2,0.5,0.4]).reshape(1,3)[0] # rotation around Z-axis (q)
alpha = np.zeros(3)                          # displacement along X-axis
a = np.array([0.75, 0.5, 0.4]).reshape(1,3)[0]               # rotation around X-axis 
revolute = [True,True,True]                  # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
tasks = [ 
            #Position2D("End-effector position", np.array([1,-1]).reshape(2,1), robot)
            #Orientation2D("End-effector orientation", np.array([-np.pi/2]).reshape(1,1), robot)
            Configuration2D("End-effector configuration", np.array([-1,-1,-np.pi/2]).reshape(3,1), robot),
            #JointPosition("Joint 1",  np.array([np.pi/2]).reshape(1,1), robot,1),
            #JointPosition("Joint 2",  np.array([0]).reshape(1,1), robot,2),
            JointPosition("Joint 3",  np.array([0]).reshape(1,1), robot,2)
        ] 

# Simulation params
dt = 1.0/60.0
Tt = 50 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy

    ### Recursive Task-Priority algorithm
    # Initialize null-space projector
    # Initialize output vector (joint velocity)
    # Loop over tasks
        # Update task state
        # Compute augmented Jacobian
        # Compute task velocity
        # Accumulate velocity
        # Update null-space projector
    ###


    null_space = np.eye(robot.dof)                  # initial null space P (projector)
    dq = np.zeros(robot.dof).reshape(-1, 1)         # initial quasi-velocities

    for i in tasks:
        i.update(robot)                             # update task Jacobian and error
        print ("i.getJacobian(): ", i.J)
        print ("null_space: ", null_space)
        J = i.getJacobian()           # task full Jacobian
        Jbar = (J @ null_space)                      # projection of task in null-space
        Jbar_inv = DLS(Jbar, 0.1)                    # pseudo-inverse or DLS
        print ("Jbar_inv: ", Jbar_inv)
        print ("j@dq: ", J@dq)
        print ("i.getError(): ", i.getError())
        dq += Jbar_inv @ ((i.getError()-J@dq))      # calculate quasi-velocities with null-space tasks execution
        null_space = null_space - np.linalg.pinv(Jbar) @ Jbar   # update null-space projector

    

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    #point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()