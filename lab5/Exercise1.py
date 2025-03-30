from lab5_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch
import time

# Robot model
d = np.zeros(3)                                         # displacement along Z-axis
theta = np.array([0.2,0.5,0.4]).reshape(1,3)[0]         # rotation around Z-axis (q)
alpha = np.zeros(3)                                     # displacement along X-axis
a = np.array([0.75, 0.5, 0.4]).reshape(1,3)[0]          # rotation around X-axis 
revolute = [True,True,True]                             # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute)       # Manipulator object

# Task hierarchy definition

# Obstacle 1
obstaclePos_1 = np.array([0.0, 1.0]).reshape(2,1)
obstacleR_1 = 0.5

# Obstacle 2
obstaclePos_2 = np.array([0.0, 0.0]).reshape(2,1)
obstacleR_2 = 0.3

# Obstacle 3
obstaclePos_3 = np.array([1.0, -0.5]).reshape(2,1)
obstacleR_3 = 0.4

obstacle_vec = []#np.array([obstaclePos_1, obstaclePos_2, obstaclePos_3])
obstacle_r = np.array([obstacleR_1, obstacleR_2, obstacleR_3])
obstacle_color = ['red', 'blue', 'green']
limits = np.array([0.5, -0.5])

# !!!!!! ensure the last task is the 2D position task
# task definition
tasks = [ 
        #Obstacle2D("Obstacle avoidance",obstaclePos_1, np.array([obstacleR_1, obstacleR_1+0.05]), robot),
        #Obstacle2D("Obstacle avoidance",obstaclePos_2, np.array([obstacleR_2, obstacleR_2+0.05]), robot),
        #Obstacle2D("Obstacle avoidance",obstaclePos_3, np.array([obstacleR_3, obstacleR_3+0.05]), robot),
        JointLimit2D("Joint limits", 1, limits, tresholds=[0.03, 0.035]),
        Position2D("End-effector position",robot, 3)
        ] 

# Simulation params
dt = 1.0/60.0
Tt = 1000
tt = np.arange(0, Tt, dt)
start_time = time.time()    # Start time

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
for i in range(len(obstacle_vec)):
    obstacle_pos = obstacle_vec[i]
    obstacle_rad = obstacle_r[i]
    ax.add_patch(patch.Circle(obstacle_pos.flatten(), obstacle_rad, color=obstacle_color[i], alpha=0.3))
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
    global start_time
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    ###

    null_space = np.eye(robot.dof)                  # initial null space P (projector)
    dq = np.zeros(robot.dof).reshape(-1, 1)         # initial quasi-velocities

    for i in tasks:
        i.update(robot)                             # update task Jacobian and error
        if i.isActive():
            """ print ("i.getJacobian(): ", i.J)
            print ("null_space: ", null_space) """
            J = i.getJacobian()           # task full Jacobian
            Jbar = (J @ null_space)                      # projection of task in null-space
            Jbar_inv = DLS(Jbar, 0.1)                    # pseudo-inverse or DLS
            """ print ("Jbar_inv: ", Jbar_inv)
            print ("j@dq: ", J@dq)
            print ("i.getError(): ", i.getError()) """
            print ("k: ", i.getK()) 
            dq += Jbar_inv @ ((i.getK()@i.getError()-J@dq) + i.ff)      # calculate quasi-velocities with null-space tasks execution
            null_space = null_space - np.linalg.pinv(Jbar) @ Jbar   # update null-space projector
    
    current_time = time.time()
    # Verify 10 sec
    if (current_time - start_time) >= 5: #or errorvec1[-1] < 0.1:
        start_time = current_time  # Reiniciar el tiempo
        tasks[-1].setRandomDesired() # the last task always has to be the 2d position

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, Tt, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot errors
plt.figure()
for i in tasks:
    if type(i) is Configuration2D:
        plt.plot(tt[:len(i.erroVec[0])], i.erroVec[0], label='Position Error')
        plt.plot(tt[:len(i.erroVec[1])], i.erroVec[1], label='Angular Error')
    elif type(i) is Obstacle2D :
        continue
    elif type(i) is JointLimit2D:
        plt.plot(tt[:len(robot.story)], robot.story, label=i.name)
        plt.axhline(y=limits[0], color='r', linestyle='--')
        plt.axhline(y=limits[1], color='r', linestyle='--')
    else:
        plt.plot(tt[:len(i.erroVec)], i.erroVec, label=i.name)

# paint line indicating limit of the angles

plt.title('Error Values Over Time')
plt.xlabel('time [s]')
plt.ylabel('Error')
plt.grid()

plt.legend()
plt.show()


