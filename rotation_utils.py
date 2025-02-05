# rotation_utils.py
import math
import numpy as np

def rotation_matrix_nd(dim, angle_deg):
    """
    Creates a block-diagonal rotation matrix for a space of dimension 'dim'
    (dim must be even) by applying the same rotation of 'angle_deg' for each pair of dimensions.
    If dim is odd, the last dimension remains unchanged.
    """
    angle = math.radians(angle_deg)
    R = np.zeros((dim, dim))
    for i in range(dim // 2):
        c = math.cos(angle)
        s = math.sin(angle)
        R[2*i, 2*i] = c
        R[2*i, 2*i+1] = -s
        R[2*i+1, 2*i] = s
        R[2*i+1, 2*i+1] = c
    if dim % 2 == 1:
        R[-1, -1] = 1.0
    return R