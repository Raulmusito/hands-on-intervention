from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model
d = np.zeros(3)                                         # displacement along Z-axis
theta = np.array([0.2,0.5,0.4]).reshape(1,3)[0]         # rotation around Z-axis (q)
alpha = np.zeros(3)                                     # displacement along X-axis
a = np.array([0.75, 0.5, 0.4]).reshape(1,3)[0]          # rotation around X-axis 
revolute = [True,True,True]                             # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute)       # Manipulator object

# Task hierarchy definition
obstacle_pos = np.array([0.0, 1.0]).reshape(2,1)
obstacle_r = 0.5

tasks = [ 
          Obstacle2D("Obstacle avoidance",obstacle_pos, np.array([obstacle_r+.05, obstacle_r+0.05]), robot),
          Position2D("End-effector position",np.array([-1.0, 1.0]).reshape(2,1),robot)
        ] 

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.add_patch(patch.Circle(obstacle_pos.flatten(), obstacle_r, color='red', alpha=0.3))
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
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    ###

    null_space = np.eye(robot.dof)                  # initial null space P (projector)
    dq = np.zeros(robot.dof).reshape(-1, 1)         # initial quasi-velocities

    for i in tasks:
        i.update(robot)                             # update task Jacobian and error
        if i.active:
            print ("i.getJacobian(): ", i.J)
            print ("null_space: ", null_space)
            J = i.getJacobian()           # task full Jacobian
            Jbar = (J @ null_space)                      # projection of task in null-space
            Jbar_inv = DLS(Jbar, 0.2)                    # pseudo-inverse or DLS
            print ("Jbar_inv: ", Jbar_inv)
            print ("j@dq: ", J@dq)
            print ("i.getError(): ", i.getError(), i.name)
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
    point.set_data(tasks[1].getDesired()[0], tasks[1].getDesired()[1])
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()