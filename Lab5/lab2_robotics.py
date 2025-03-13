import numpy as np # Import Numpy 

def wrapangle (angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
    
    #  Displacement along Z-axis
    dis_z=np.array([1,0,0,0,0,1,0,0,0,0,1,d,0,0,0,1]).reshape(4,4) 
    # Totation around Z-axis
    rot_z=np.array([np.cos(theta),-np.sin(theta),0,0,np.sin(theta),np.cos(theta),0,0,0,0,1,0,0,0,0,1]).reshape(4,4)
    # Displacement along X-axis
    dis_x=np.array([1,0,0,a,0,1,0,0,0,0,1,0,0,0,0,1]).reshape(4,4)
    # Rotation around X-axis
    rot_x=np.array([1,0,0,0,0,np.cos(alpha),-np.sin(alpha),0,0,np.sin(alpha),np.cos(alpha),0,0,0,0,1]).reshape(4,4)
    
    # Multiply the matrices by the right to indicate the correct rotation - translation order
    T = dis_z@rot_z@dis_x@rot_x

    return T # Return the transformation matrix

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

    for i in range(len(d)): # Compute the DH transformation for each joint 
        new = T[-1]@DH(d[i], theta[i], a[i], alpha[i]) # Compute the accumulated transformation
        T.append(new) # Append the new transformation
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

    # Build the jacobian matrix
    J = np.zeros((6,len(revolute))) 
    O = T[-1][0:3,3] # End-effector position

    for i in range(len(revolute)): # For each joint
        z = T[i][0:3,2] # Extract z
        Oi = T[i][0:3,3] # Extract o

        # Check joint type
        if revolute[i]:
            J[:,i] = np.concatenate((np.cross(z, O - Oi), z)) # Build the jacobian matrix by asseigassigning the columns
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
    # WARNING!!!! the damping factor change a lot the performance of the algorithm
    # Maybe it is worth implementing the variable damping factor
    # yes, I wrote this, is not AI :(

    I = np.eye(A.shape[0])  # Identity matrix matching columns of A

    # Compute the DLS 
    interior = A @ A.T + damping**2 * I
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

