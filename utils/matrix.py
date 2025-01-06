import numpy as np

def quaternion_to_rotation_vectorized(q):
    """
    Convert batch of quaternions to rotation matrices
    Input: q of shape (N, 4) where N is batch size
    Output: rotation matrices of shape (N, 3, 3)
    """
    # Unpack quaternions
    q_r = q[..., 0]
    q_i = q[..., 1]
    q_j = q[..., 2]
    q_k = q[..., 3]
    
    # Preallocate rotation matrices
    r = np.zeros((*q.shape[:-1], 3, 3))
    
    # Fill the rotation matrices
    r[..., 0, 0] = 1 - 2*(q_j**2 + q_k**2)
    r[..., 0, 1] = 2*(q_i*q_j - q_r*q_k)
    r[..., 0, 2] = 2*(q_i*q_k + q_r*q_j)
    r[..., 1, 0] = 2*(q_i*q_j + q_r*q_k)
    r[..., 1, 1] = 1 - 2*(q_i**2 + q_k**2)
    r[..., 1, 2] = 2*(q_j*q_k - q_r*q_i)
    r[..., 2, 0] = 2*(q_i*q_k - q_r*q_j)
    r[..., 2, 1] = 2*(q_j*q_k + q_r*q_i)
    r[..., 2, 2] = 1 - 2*(q_i**2 + q_j**2)
    
    return r