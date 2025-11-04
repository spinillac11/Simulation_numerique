import numpy as np

position = np.array([[2., 2.], [5., 2.], [2., 5.], [5., 5.]])
velocity = 2*np.random.normal(size=position.shape)
sigma = 1
L = 7

print("velocity:", velocity)

collision_time = np.where(velocity > 0, (L-sigma-position)/velocity, (sigma-position)/velocity)
first_collision_time = np.min(collision_time)
where_par = np.where(collision_time == first_collision_time)[0][0]
where_x = np.where(collision_time == first_collision_time)[1][0]
particule = where_par

print(where_par)
print(where_x)

print(velocity[where_par][where_x])


print("collition time:",  collision_time)
print("First_collition time: ", first_collision_time)

