import numpy as np # Import Numpy 

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    
    # D is the displacement along Z-axis
    dis_z=np.array([1,0,0,0,0,1,0,0,0,0,1,d,0,0,0,1]).reshape(4,4) 
    # Th is the rotation around Z-axis
    rot_z=np.array([np.cos(theta),-np.sin(theta),0,0,np.sin(theta),np.cos(theta),0,0,0,0,1,0,0,0,0,1]).reshape(4,4)
    # A is the displacement along X-axis
    dis_x=np.array([1,0,0,a,0,1,0,0,0,0,1,0,0,0,0,1]).reshape(4,4)
    # Al is the rotation around X-axis
    rot_x=np.array([1,0,0,0,0,np.cos(alpha),-np.sin(alpha),0,0,np.sin(alpha),np.cos(alpha),0,0,0,0,1]).reshape(4,4)
    
    # 2. Multiply matrices in the correct order (result in T).
    T = dis_z@rot_z@dis_x@rot_x
    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.

    for i in range(len(d)):
        new = T[-1]@DH(d[i], theta[i], a[i], alpha[i])
        T.append(new)
    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.

    """     z0 = np.array([0,0,1]).T
    o0 = np.array([0,0,0]).T
    on = T[-1][0:3,3]

    J = np.array([])
    for i in range(len(T[1:])):
        if i == 0:
            z = z0
            o = o0
        else:
            z = T[i][0:3,2]
            o = T[i][0:3,3]
        up = revolute[i] * np.cross(z,(on - o))+(1-revolute[i])*z
        down = revolute[i] * z

        j2 = np.array([up[0],up[1],up[2],down[0],down[1],down[2]]).reshape(6,1)
        if J.size == 0:
            J = j2
        else:
            J =  np.hstack((J, j2))
    return J """
    J = np.zeros((6,len(revolute)))
    O = T[-1][0:3,3]
    for i in range(len(revolute)):
        z = T[i][0:3,2]
        Oi = T[i][0:3,3]
        if revolute[i]:
            J[:,i] = np.concatenate((np.cross(z, O - Oi), z))
        else:
            J[:,i] = np.concatenate((z, np.zeros(3)))
    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    I = np.eye(6)  # Identity matrix matching columns of A
    interior = A @ A.T + damping**2 * I
    print (type(interior))
    dls =  A.T @np.linalg.inv( interior  )
    return dls # Implement the formula to compute the DLS of matrix A.

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P

DH(4,5,6,7)