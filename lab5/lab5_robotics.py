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
        self.story = []
        self.update(0.0, 0.0)


    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        self.story.append(self.q[0][0])
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)
        print ("self.t", self.T)
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
    
    def getLINKJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)

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
    
    def getLinkTransform(self, link):
        return self.T[link]
        

    def getLinkOrientation(self, link):
        linkT = self.getLinkTransform(link)
        return np.array(np.arctan2(linkT[1,0], linkT[0,0])).reshape((1,1))

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
        self.erroVec = [] # error vector
        #self.l
        self.ff = None
        self.k = None

    def getFF(self):
        return self.ff
    
    def setFF(self, ff):
        self.ff = ff

    def getK(self):
        return self.k

    def setK(self, k):
        self.k = k
 
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
    
    """
        Method that activation state of the task.
    """
    def isActive (self):
        return self.active

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, robot: Manipulator, link):
        super().__init__(name, self.setRandomDesired())
        self.J = np.zeros((2,robot.getDOF()))                  # Initialize with proper dimensions
        self.err = np.zeros((2,1))                                  # Initialize with proper dimensions
        self.setK(np.eye(2))
        self.setFF(np.zeros((2,1)))
        self.link = link
        self.active = True
        
    def update(self, robot: Manipulator):
        self.J = robot.getLINKJacobian(self.link)[:2,:].reshape((2,self.link))                     # Update task Jacobian
        self.J = np.hstack((self.J, np.zeros((2, robot.dof - self.link))))
        self.err = np.array(self.getDesired() - robot.getLinkTransform(self.link)[0:2,3].reshape((2,1))) # Update task error
        self.erroVec.append(np.linalg.norm(self.err))

    def setRandomDesired(self):
        random = (np.random.rand(2,1)*2-1).reshape((2,1))
        self.setDesired(random)
        return random

'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, robot: Manipulator, link):
        super().__init__(name, desired)
        self.J = np.zeros((1,robot.getDOF()))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.setK(np.eye(1))
        self.setFF(np.zeros((1,1)))
        self.link = link

        
    def update(self, robot: Manipulator):
        self.J = robot.getLINKJacobian(self.link)[5,:].reshape((1,self.link))   # Update task Jacobian
        self.J = np.pad(self.J, (0, robot.dof - self.link), mode='constant', constant_values=0)
        current_transform = robot.getLinkTransform(self.link) # Compute current sigma
        current_sigma = np.array(np.arctan2(current_transform[1,0], current_transform[0,0])).reshape((1,1)) # Compute current sigma
        print('current_sigma:',current_sigma)
        self.err = wrapangle(self.getDesired() - current_sigma.reshape((1,1))) # Update task error
        print ("angular error: ", self.err)
        self.erroVec.append(self.err[0])
        pass # to remove

    def setRandomDesired(self):
        self.setDesired( (np.random.rand(1,1)*2*np.pi-np.pi).reshape((1,1)))
        self.setDesired(np.array([np.pi]).reshape(1,1))
        pass
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, robot: Manipulator, link):
        super().__init__(name, desired)
        self.J = np.zeros((3,robot.getDOF()))# Initialize with proper dimensions
        self.err = np.zeros((3,1))# Initialize with proper dimensions
        self.erroVec = [[],[]]
        self.setK(np.eye(3))
        self.setFF(np.zeros((3,1)))
        self.link = link


        
    def update(self, robot: Manipulator):
        positionJacobian = robot.getLINKJacobian(self.link)[:2,:].reshape((2,self.link))                     # Update task Jacobian
        positionJacobian = np.hstack((positionJacobian, np.zeros((2, robot.dof - self.link))))
        self.J[0:2,:] = positionJacobian # Update task Jacobian

        orientationJacobian = robot.getLINKJacobian(self.link)[5,:].reshape((1,self.link))   # Update task Jacobian
        orientationJacobian =  np.hstack((orientationJacobian, np.zeros((1, robot.dof - self.link))))
        self.J[2,:] = orientationJacobian

        current_transform = robot.getLinkTransform(self.link) # Compute current sigma
        current_sigma_angle = np.arctan2(current_transform[1,0], current_transform[0,0]) # Compute current sigma angle
        current_sigma_pos = current_transform[0:2,3] # Compute current sigma position
        error_pos = self.getDesired()[0:2] - current_sigma_pos.reshape((2,1)) # Compute position error
        error_angle = self.getDesired()[2] - current_sigma_angle
        print ("angular error: ", error_angle)
        self.err = np.array([error_pos[0], error_pos[1], error_angle]).reshape((3,1)) # Update task error
        self.erroVec[0].append(np.linalg.norm(error_pos))
        self.erroVec[1].append(error_angle[0])
        pass
    def setRandomDesired(self):
        self.setDesired(np.array([np.random.rand(1,1)*2-1,np.random.rand(1,1)*2-1, np.random.rand(1,1)*2*np.pi-np.pi]).reshape((3,1)))
        pass
        
''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired, robot: Manipulator, link):
        super().__init__(name, desired)
        self.link = link
        self.J = np.zeros((1,self.link))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.setK(np.eye(1))
        self.setFF(np.zeros((1,1)))


        
    def update(self, robot: Manipulator):
        self.J = robot.getLINKJacobian(self.link)[5,:].reshape((1,self.link))   # Update task Jacobian
        self.J = np.pad(self.J, (0, robot.dof - self.link), mode='constant', constant_values=0)
        current_sigma = robot.getLinkOrientation(self.link) # Compute current sigma
        print('current_sigma:',current_sigma)
        self.err = wrapangle(self.getDesired() - current_sigma.reshape((1,1))) # Update task error
        print ("angular error: ", self.err)
        self.erroVec.append(self.err[0])
        pass # to remove

    def setRandomDesired(self):
        self.setDesired( (np.random.rand(1,1)*2*np.pi-np.pi).reshape((1,1)))
        pass

''' 
    Subclass of Task, representing the Obstacle avoidance task.
'''
class Obstacle2D(Task):
    def __init__(self, name, position, thresholds, robot: Manipulator,):
        super().__init__(name, None)
        self.position = position
        self.activation_tresh = thresholds[0]
        self.deactivation_tresh = thresholds[1]
        self.J = np.zeros((2,robot.dof))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.active = False
        self.setK(np.eye(2))
        self.setFF(np.zeros((2,1)))
    
    def activate (self, sigma):
        if self.active == False and np.linalg.norm(sigma) <= self.activation_tresh:
            self.active = True
        elif self.active == True and np.linalg.norm(sigma) >= self.deactivation_tresh:
            self.active = False

    def update(self, robot: Manipulator):
        self.J = robot.getEEJacobian()[:2,:].reshape((2,robot.dof))   # Update task Jacobian
        current_sigma = robot.getEETransform()[:2,3].reshape((2,1)) - self.position#g get EE x & y
        self.activate(current_sigma)
        print("current_sigma: ",current_sigma)
        self.err = current_sigma/np.linalg.norm(current_sigma)
        print ("angular error: ", self.err)
        pass # to remove

class JointLimit2D(Task)    :
    def __init__(self, name, link, limits, tresholds):
        super().__init__(name, None)
        self.link = link
        self.limits = limits
        self.activation_tresh = tresholds[0]
        self.deactivation_tresh = tresholds[1]
        self.J = np.zeros((1,self.link))# Initialize with proper dimensions
        self.err = np.zeros((1,1))# Initialize with proper dimensions
        self.setK(np.eye(1))
        self.setFF(np.zeros((1,1)))
        self.active = 0

    def update(self, robot: Manipulator):
        self.J = robot.getLINKJacobian(self.link)[5,:].reshape((1,self.link))   # Update task Jacobian
        self.J = np.pad(self.J, (0, robot.dof - self.link), mode='constant', constant_values=0)
        current_sigma = robot.getLinkOrientation(self.link) # Compute current sigma
        self.activate(current_sigma)
        print('current_sigma:',current_sigma)
        self.err = np.array([1*self.active]) # Update task error
        print ("angular error: ", self.err)
        self.erroVec.append(self.err)
        pass # to remove

    
    def activate (self, angle):
        if self.active == 0 and angle >=  self.limits[0] - self.activation_tresh:
            self.active = -1
        elif self.active == 0 and angle <= self.limits[1] + self.activation_tresh:
            self.active = 1
        elif self.active == -1 and angle <= self.limits[0] - self.deactivation_tresh:
            self.active = 0
        elif self.active == 1 and angle >= self.limits[1] + self.deactivation_tresh:
            self.active = 0
    
    def isActive(self):
        return abs(self.active)


    def setRandomDesired(self):
        self.setDesired( (np.random.rand(1,1)*2*np.pi-np.pi).reshape((1,1)))
        pass