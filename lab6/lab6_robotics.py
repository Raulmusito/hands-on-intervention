from lab5_robotics import *
import math as m

class MobileManipulator:
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
        self.revoluteExt = [True, False, True, True, True]    # List of joint types extended with base joints
        self.r = 0            # Distance from robot centre to manipulator base
        self.dof = len(self.revoluteExt) # Number of DOF of the system
        self.q = np.zeros((len(self.revolute),1)) # Vector of joint positions (manipulator)
        self.eta = np.zeros((3,1)) # Vector of base pose (position & orientation)
        self.story = []
        self.story_base = []
        self.update(np.zeros((self.dof,1)), 0.0) # Initialise robot state

    '''
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    '''
    def update(self, dQ, dt, method = 3 ):
        # Update manipulator
        self.story.append(-self.q[2][0] )
        self.story_base.append(self.eta[0][0])
        self.q += dQ[2:, 0].reshape(-1,1) * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]

        # Update mobile base pose
        
        v = dQ[1,0] # get linear velocity
        w = dQ[0,0] # get angular velocity


        if method == 1:
            """First move forward then rotate"""
            self.eta[0,0] += v * dt * np.cos(self.eta[2,0])
            self.eta[1,0] += v * dt * np.sin(self.eta[2,0])
            self.eta[2,0] += w * dt

        elif method == 2:
            """First rotate then move forward"""
            self.eta[2,0] += w * dt
            self.eta[0,0] += v * dt * np.cos(self.eta[2,0])
            self.eta[1,0] += v * dt * np.sin(self.eta[2,0])

        elif method == 3:
            """Move forward and rotate at the same time"""
            # to prevent division by zero
            if abs(w) < 0.000001:
                self.eta[0,0] += v * dt * np.cos(self.eta[2,0])
                self.eta[1,0] += v * dt * np.sin(self.eta[2,0])
            
            else:
                R = v / w  # Radius of the circular path
                inc_w = w * dt  # Change in angle
                self.eta[0,0] += R * (np.sin(self.eta[2,0] + inc_w) - np.sin(self.eta[2,0]))
                self.eta[1,0] -= R * (np.cos(self.eta[2,0] + inc_w) - np.cos(self.eta[2,0]))
                self.eta[2,0] += inc_w

        # Base kinematics
        Tb = np.eye(4)
        Tb[0:2, 0:2] = np.array([[np.cos(self.eta[2,0]), -np.sin(self.eta[2,0])],
                                [np.sin(self.eta[2,0]), np.cos(self.eta[2,0])]])
        Tb[0:2, 3] = self.eta[0:2, 0]

        # Combined the system kinematics (DH parameters extended with base DOF)
        self.theta += -np.pi/2.0    
        dExt = np.concatenate([np.array([0, self.r]), self.d])
        thetaExt = np.concatenate([np.array([np.pi/2.0 ,0.0]), self.theta])
        aExt = np.concatenate([np.array([0, 0.0]), self.a])
        alphaExt = np.concatenate([np.array([np.pi/2.0, -np.pi/2.0]), self.alpha])

        self.T = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)


    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

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
        return self.q[joint-2]


    def getBasePose(self):
        return self.eta
    
    def getEEPose(self):
        return self.T[-1][0:2,3].reshape(2,1)

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

    ###
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revoluteExt, link)

    def getLinkTransform(self, link):
        return self.T[link]
    
    """ def getLinkOrientation(self, link):
        linkT = self.getLinkTransform(link)
        return np.array(np.arctan2(linkT[1,0], linkT[0,0])).reshape((1,1)) """

    
    def getLinkOrientation(self, link):
        linkT = self.getLinkTransform(link)
        linkBase = self.getLinkTransform(0)
        base_anlge = wrapangle(np.arctan2(linkBase[1,0], linkBase[0,0]))
        joint_angle = wrapangle(np.arctan2(linkT[1,0], linkT[0,0]))
        return np.array(wrapangle(joint_angle - base_anlge)).reshape((1,1))

