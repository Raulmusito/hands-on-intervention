from lab2_robotics import * # Includes numpy import


def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # Code almost identical to the one from lab2_robotics...
    # 1. Initialize J and O.
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.

    # Build the jacobian matrix
    J = np.zeros((6, link)) 
    O = T[-1][0:3,3] # End-effector position

    for i in range(link): # For each joint
        z = T[i][0:3,2] # Extract z
        Oi = T[i][0:3,3] # Extract o

        # Check joint type
        if revolute[i]:
            J[:,i] = np.concatenate((np.cross(z, O - Oi), z)) # Build the jacobian matrix by asseigassigning the columns
        else:
            J[:,i] = np.concatenate((z, np.zeros(3)))
    return J

'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.array(theta).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, robot: Manipulator):
        super().__init__(name, desired)
        self.J = np.zeros((2,robot.getDOF()))                  # Initialize with proper dimensions
        self.err = np.zeros((2,1))                                  # Initialize with proper dimensions
        
        
    def update(self, robot: Manipulator):
        self.J = robot.getEEJacobian()[:2,:]                     # Update task Jacobian
        self.err = np.array(self.getDesired() - robot.getEETransform()[0:2,3].reshape((2,1))) # Update task error
        pass # to remove
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, robot: Manipulator):
        super().__init__(name, desired)
        self.J = np.zeros((2,robot.getDOF()))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        
    def update(self, robot: Manipulator):
        self.J = robot.getEEJacobian()[:2,:]   # Update task Jacobian
        current_sigma = np.array(np.arctan2(robot.getEETransform()[1,0], robot.getEETransform()[0,0])).reshape((1,1)) # Compute current sigma
        self.err = wrapangle(self.getDesired() - current_sigma.reshape((1,1))) # Update task error
        print ("angular error: ", self.err)
        pass # to remove
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, robot: Manipulator):
        super().__init__(name, desired)
        self.J = np.zeros((3,robot.getDOF()))# Initialize with proper dimensions
        self.err = np.zeros((3,1))# Initialize with proper dimensions
        
    def update(self, robot: Manipulator):
        self.J = robot.getEEJacobian()[:3,:] # Update task Jacobian
        current_sigma_angle = np.arctan2(robot.getEETransform()[1,0], robot.getEETransform()[0,0]) # Compute current sigma angle
        current_sigma_pos = robot.getEETransform()[0:2,3] # Compute current sigma position

        error_pos = self.getDesired()[0:2] - current_sigma_pos.reshape((2,1)) # Compute position error
        error_angle = self.getDesired()[2] - current_sigma_angle
        print ("angular error: ", error_angle)
        self.err = np.array([error_pos[0], error_pos[1], error_angle]).reshape((3,1)) # Update task error
        pass
        
''' 
    Subclass of Task, representing the joint position task.
'''
# class JointPosition(Task):
