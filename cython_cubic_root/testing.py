import cubic_root_closest_to_0
import numpy as np
import cube_solver

d1 = 2
d2 = 3

x = np.array(list(range(4 * d1 * d2))).reshape((4, d1, d2)).astype('float32') + 1
print(x.dtype)

setup = """
import cubic_root_closest_to_0
import numpy as np
import cube_solver

d1 = 20
d2 = 30

x = np.array(list(range(4 * d1 * d2))).reshape((4, d1, d2)).astype('float32') + 1
"""

# Test that both methods give the same roots
print(cubic_root_closest_to_0.get_roots(x))
print(np.apply_along_axis(cube_solver.real_root_closest_to_zero, 0, x))

# Time each method
import timeit
number_of_times = 1000

cython_time = timeit.timeit(setup=setup,
                            stmt='cubic_root_closest_to_0.get_roots(x)',
                            number=number_of_times)
print('cython_time:\t', cython_time)

np_time = timeit.timeit(setup=setup,
                        stmt='np.apply_along_axis(cube_solver.real_root_closest_to_zero, 0, x)',
                        number=number_of_times)
print('np_time:\t\t', np_time)
