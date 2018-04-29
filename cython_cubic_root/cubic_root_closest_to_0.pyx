import numpy as np
import cube_solver


cpdef get_roots(float [:,:,:] x):

    # Calculate shape o
    cdef int b = x.shape[1]  # Mini-batch size
    cdef int m = x.shape[2]  # Output size from neural network layer

    u_np =  np.zeros([b, m], dtype='float32')
    cdef float [:,:] u = u_np

    for i in range(b):
        for j in range(m):
            u[i, j] = cube_solver.real_root_closest_to_zero(x[:, i, j])

    return u_np