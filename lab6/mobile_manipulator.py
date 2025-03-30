from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans
import time

# Robot model
d = np.zeros(3)                                  # displacement along Z-axis
theta = np.array([0.2,0.5,0.4]).reshape(1,3)[0]  # rotation around Z-axis
alpha = np.zeros(3)                              # rotation around X-axis
a =  np.array([0.75, 0.5, 0.4]).reshape(1,3)[0]  # displacement along X-axis
revolute = [True, True, True]                     # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Task definition
limits = np.array([np.pi/2, -np.pi/2])

desiredVector = [np.array([-1,-1,0]).reshape(3,1),
                 np.array([ -1, 1, np.pi/2]).reshape(3,1),
                 np.array([1, 1, -np.pi      ]).reshape(3,1),
                 np.array([ 1,-1,-np.pi/2]).reshape(3,1),
                 np.array([-1, 1, 0      ]).reshape(3,1)]

tasks = [
        Configuration2D("End-effector configuration", robot,5, desiredVector),
        #JointLimit2D("Joint limits", 3, limits, tresholds=[0.03, 0.03]),
        #Position2D("End-effector position", robot, 5),
        #Orientation2D("End-effector orientation", np.array([np.pi]).reshape(1,1), robot,5)
        ] 

# Simulation params
dt = 1.0/60.0
Tt = 60
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
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

base_history = []
ee_history = []
velocity_history = []

""" 
    Method to update the simulation
    1 = first move forward then rotate
    2 = first rotate then move forward
    3 = move forward and rotate at the same time 
"""
method = 3

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
    global base_history, ee_history
    global method
    
    ### Recursive Task-Priority algorithm
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    ###

    # Initialize variables
    null_space = np.eye(robot.dof)                  # initial null space P (projector)
    dq = np.zeros(robot.dof).reshape(-1, 1)         # initial quasi-velocities

    base_history.append(robot.eta[0:2,0].flatten())
    ee_history.append(robot.getEEPose()[0:2,0].flatten())
    velocity_history.append(dq)

    for i in tasks:
        i.update(robot)                             # update task Jacobian and error
        if i.isActive():
            """ print ("i.getJacobian(): ", i.J)
            print ("null_space: ", null_space) """
            J = i.getJacobian()           # task full Jacobian
            Jbar = (J @ null_space)                      # projection of task in null-space
            DLS_weights = np.diag([2, 2, 0.5, 0.5, 0.5]) # weights for DLS
            Jbar_inv = DLS(Jbar, 0.05, DLS_weights)                    # pseudo-inverse or DLS
            """ print ("Jbar_inv: ", Jbar_inv)
            print ("j@dq: ", J@dq)
            print ("i.getError(): ", i.getError()) 
            print ("k: ", i.getK()) """
            dq += Jbar_inv @ ((i.getK()@i.getError()-J@dq) + i.ff)      # calculate quasi-velocities with null-space tasks execution
            null_space = null_space - np.linalg.pinv(Jbar) @ Jbar   # update null-space projector
    current_time = time.time()
    # Verify 10 sec
    if (current_time - start_time) >= 5: #or errorvec1[-1] < 0.1:
        start_time = current_time  # Reiniciar el tiempo
        tasks[-1].setRandomDesired()#random = 1) # the last task always has to be the 2d position
    # Update robot
    robot.update(dq, dt, method)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    for i in tasks:
        if type(i) == Position2D or type(i) == Configuration2D:
            # Update the end-effector position
            point.set_data(i.getDesired()[0], i.getDesired()[1])

    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate,tt, 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()


# Plot errors


plt.figure(figsize=(10, 8))

# First subplot: Error Values Over Time
plt.subplot(2, 1, 1)
for i in tasks:
    if isinstance(i, Configuration2D):
        plt.plot(tt[:len(i.erroVec[0])], i.erroVec[0], label='Position Error')
        plt.plot(tt[:len(i.erroVec[1])], i.erroVec[1], label='Angular Error')
    elif isinstance(i, Obstacle2D):
        continue
    elif isinstance(i, JointLimit2D):
        plt.plot(tt[:len(robot.story)], robot.story, label=i.name)
    else:
        plt.plot(tt[:len(i.erroVec)], i.erroVec, label=i.name)

plt.title('Error Values Over Time')
plt.xlabel('Time [s]')
plt.ylabel('Error')
plt.grid()
plt.legend()

# Second subplot: Velocity History
plt.subplot(2, 1, 2)
velocity_history = np.array(velocity_history)
plt.plot(tt[:len(velocity_history)], velocity_history[:, 0], label='Angular Velocity')
plt.plot(tt[:len(velocity_history)], velocity_history[:, 1], label='Linear Velocity')
plt.plot(tt[:len(velocity_history)], velocity_history[:, 2], label='Joint 1 Velocity')
plt.plot(tt[:len(velocity_history)], velocity_history[:, 3], label='Joint 2 Velocity')
plt.plot(tt[:len(velocity_history)], velocity_history[:, 4], label='Joint 3 Velocity')

plt.title('Velocity History')
plt.xlabel('Time [s]')
plt.ylabel('Velocity')
plt.grid()
plt.legend()

# Adjust layout and show
plt.tight_layout()
plt.show()


""" # Plot base and end-effector history
plt.figure()
plt.plot(*zip(*base_history), label='Base')
plt.plot(*zip(*ee_history), label='End-effector')
plt.title('Base and End-effector History')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.grid()
plt.legend()
plt.show() """

""" # save the histories
np.save('base_history2'+str(method), base_history)
np.save('ee_history2'+str(method), ee_history) """

