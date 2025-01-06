import numpy as np

# -------- QUATERNION TO ROTATION --------

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

# -------- PROJECT POINTS --------

def project_points(points, e, K, dist_coeffs=None):
    # homogenous coordinates
    points_h = np.hstack([points, np.ones((len(points), 1))])
    # First apply extrinsics (4x4 @ 4xN)
    transformed_points = (e @ points_h.T)
    # Then get normalized coordinates (before K)
    transformed_points = transformed_points[:3, :]
    normalized_coords = transformed_points / transformed_points[2:3, :]  # Divide by Z
    x, y = normalized_coords[0], normalized_coords[1]

    if dist_coeffs is not None:
        # Apply distortion
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2

        # Radial distortion
        k1, k2, k3 = dist_coeffs[:3]
        radial = (1 + k1*r2 + k2*r4 + k3*r6)
        
        xd = x * radial
        yd = y * radial

        # Convert back to homogeneous coordinates
        normalized_coords = np.vstack((xd, yd, np.ones_like(x)))

    # Now project with intrinsics
    projected_points = (K @ normalized_coords).T
    # Already in image coordinates since we applied distortion to normalized coordinates
    projected_points = projected_points[:, :2]

    return projected_points

def select_points_within_bounds(projected_points, width, height, colors=None):
    bound_indices = np.all(projected_points >= 0, axis=1) & np.all(projected_points < [width, height], axis=1)
    bound_points = projected_points[bound_indices]
    if colors is not None:
        bound_colors = colors[bound_indices]
        return bound_points, bound_colors
    return bound_points, None