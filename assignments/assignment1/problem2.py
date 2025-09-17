import os
import numpy as np
import mengine as m
np.set_printoptions(precision=3, suppress=True)

# NOTE: This assignment asks you to test rotations and translations in varying order.

# Create environment and ground plane
env = m.Env()
ground = m.Ground([0, 0, -0.5])
env.set_gui_camera(look_at_pos=[0, 0, 0])

# position definition
pos = np.array([0.2, 0, 0.0])
# orientation definition as rotation matrix
orient = np.eye(3)

# Create box to transform
box = m.Shape(m.Box(half_extents=[0.1, 0.2, 0.05]), static=True,
              position=pos, orientation=m.get_quaternion(orient), rgba=[0, 1, 0, 1])

def euler_to_rotation_matrix(alpha, beta, gamma):
    # Convert euler angles (alpha, beta, gamma) to rotation matrix
    # input: alpha, beta, gamma: euler angles
    # output: R: rotation matrix

    # ------ TODO Student answer below -------
    #Note: Using ZYZ Euler Angle convention with intrinsic rotations:
    def Rz(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    def Ry(theta):
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)

    # ------ Student answer above -------

def apply_transform(pos, orient, d, euler):
    # transform a box using translation d and rotation given by euler angles
    # input: pos, orient: current position and orientation (rotation matrix) of the box
    #        d: target translation
    #        euler: target rotation in euler angles
    # output: pos_new, orient_new: new position and orientation (rotation matrix) of the box
    # ------ TODO Student answer below -------
    
    #Create homogeneous 4x4 transform of current and transform poses

    T_current = np.eye(4)
    #populate matrix
    T_current[0:3,0:3] = orient
    T_current[0:3,3] = pos

    T = np.eye(4)
    T[0:3,0:3] = euler_to_rotation_matrix(euler[0], euler[1], euler[2])
    T[0:3,3] = d

    #Apply Transform to T_current
    T_new = T @ T_current

    #Return variables
    pos_new = T_new[0:3,3]
    orient_new = T_new[0:3,0:3]

    return pos_new, orient_new
    # return np.zeros(3), np.eye(3)
    # ------ Student answer above -------

def wait_for_enter():
    # NOTE: Press enter to continue to next angles
    print('Press enter in the simulator to continue to the next angle set')
    keys = m.get_keys()
    while True:
        keys = m.get_keys()
        if 'return' in keys:
            break
        m.step_simulation(realtime=True)
    m.step_simulation(steps=50, realtime=True)

# Test cases for rotations
# T_1, T2, T3
d_1 = np.array([0.1, -0.2, 0.05])
d_2 = np.array([-0.01, 0.04, 0.2])
d_3 = np.array([0.0, 0.0, 0.0])

euler_1 = np.radians([20, 45, 10])
euler_2 = np.radians([-15, 7, 23])
euler_3 = np.radians([65, 21, -19])

# 1: T1, T2, T3
pos1, orient1 = apply_transform(pos, orient, d_1, euler_1)
pos2, orient2 = apply_transform(pos1, orient1, d_2, euler_2)
pos_final, orient_final = apply_transform(pos2, orient2, d_3, euler_3)
# update box position and orientation
box.set_base_pos_orient(pos_final, m.get_quaternion(orient_final))
print('-'*20)
print('final position:', pos_final)
print('final orientation:', orient_final)
print('-'*20)

# NOTE: Press enter to continue to next angles
wait_for_enter()

# 2: T2, T1, T3
pos1, orient1 = apply_transform(pos, orient, d_2, euler_2)
pos2, orient2 = apply_transform(pos1, orient1, d_1, euler_1)
pos_final, orient_final = apply_transform(pos2, orient2, d_3, euler_3)
# update box position and orientation
box.set_base_pos_orient(pos_final, m.get_quaternion(orient_final))
print('-'*20)
print('final position:', pos_final)
print('final orientation:', orient_final)
print('-'*20)

# NOTE: Press enter to continue to next angles
wait_for_enter()

# 3: T3, T2, T1
pos1, orient1 = apply_transform(pos, orient, d_3, euler_3)
pos2, orient2 = apply_transform(pos1, orient1, d_2, euler_2)
pos_final, orient_final = apply_transform(pos2, orient2, d_1, euler_1)
# update box position and orientation
box.set_base_pos_orient(pos_final, m.get_quaternion(orient_final))
print('-'*20)
print('final position:', pos_final)
print('final orientation:', orient_final)
print('-'*20)

# NOTE: Press enter to continue to next angles
wait_for_enter()

