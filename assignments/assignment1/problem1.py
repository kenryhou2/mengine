import os
import numpy as np
import mengine as m
from scipy.spatial.transform import Rotation as Rot # For ground truth checking
np.set_printoptions(precision=3, suppress=True)

# NOTE: This problem asks you to convert between the different rotation representations.

# Create environment and ground plane
env = m.Env()
ground = m.Ground([0, 0, -0.5])
env.set_gui_camera(look_at_pos=[0, 0, 0])

# position definition
x = np.array([0.2, 0, 0])

# Create points to rotate
# point rotated using euler angles
point_e = m.Shape(m.Sphere(radius=0.03), static=True,
                  position=x, rgba=[0, 1, 0, 0.2])
# point rotated using axis-angle
point_aa = m.Shape(m.Sphere(radius=0.025), static=True,
                   position=x, rgba=[1, 0, 0, 0.2])
# point rotated using rotation matrix
point_r = m.Shape(m.Sphere(radius=0.02), static=True,
                  position=x, rgba=[0, 0, 1, 0.2])


def rodrigues_formula(n, x, theta):
    # Rodrigues' formula for axis-angle: rotate a point x around an axis n by angle theta
    # input: n, x, theta: axis, point, angle
    # output: x_new: new point after rotation
    # ------ TODO Student answer below -------
    #normalize the axis vector
    n = n / np.linalg.norm(n)
    x_rot = x * np.cos(theta) + np.cross(n, x) * np.sin(theta) + n * (np.dot(n, x)) * (1 - np.cos(theta))
    return x_rot
    
    # return np.zeros(3)
    # ------ Student answer above -------


def rotate_euler(alpha, beta, gamma, x):
    # Rotate a point x using euler angles (alpha, beta, gamma)
    # input: alpha, beta, gamma: euler angles
    # output: x_new: new point after rotation

    # ------ TODO Student answer below -------
    R = euler_to_rotation_matrix(alpha, beta, gamma)
    return R @ x

    # ------ Student answer above -------


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


def euler_to_axis_angle(alpha, beta, gamma):
    # Convert euler angles (alpha, beta, gamma) to axis-angle representation (n, theta)
    # input: alpha, beta, gamma: euler angles
    # output: n, theta
    # ------ TODO Student answer below -------
    def rotmat_to_quat(R, eps=1e-12):
        """
        Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].
        Follows the convention: x,y,z are the vector part, w is scalar.
        """
        R = np.asarray(R, dtype=float)
        assert R.shape == (3, 3)

        trace = np.trace(R)
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        else:
            # Find the major diagonal element among R[0,0], R[1,1], R[2,2]
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(max(eps, 1.0 + R[0,0] - R[1,1] - R[2,2]))
                w = (R[2,1] - R[1,2]) / s
                x = 0.25 * s
                y = (R[0,1] + R[1,0]) / s
                z = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(max(eps, 1.0 + R[1,1] - R[0,0] - R[2,2]))
                w = (R[0,2] - R[2,0]) / s
                x = (R[0,1] + R[1,0]) / s
                y = 0.25 * s
                z = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(max(eps, 1.0 + R[2,2] - R[0,0] - R[1,1]))
                w = (R[1,0] - R[0,1]) / s
                x = (R[0,2] + R[2,0]) / s
                y = (R[1,2] + R[2,1]) / s
                z = 0.25 * s

        q = np.array([x, y, z, w])
        return q / (np.linalg.norm(q) + eps)
    R = euler_to_rotation_matrix(alpha, beta, gamma)
    q = rotmat_to_quat(R)
    theta = 2 * np.arccos(q[3])
    s = np.sqrt(1 - q[3]**2)
    if s < 1e-12:
        axis = np.array([1, 0, 0])  # Arbitrary axis
    else:
        axis = q[0:3] / s
    return axis, theta
    # ------ Student answer above -------


x_new_e = np.array([0.2, 0, 0])
x_new_r = np.array([0.2, 0, 0])
x_new_aa = np.array([0.2, 0, 0])

for alpha, beta, gamma in zip([20, -25, 0], [45, 5, 135], [10, 90, -72]):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    (n, theta) = euler_to_axis_angle(alpha, beta, gamma)
    
    R = euler_to_rotation_matrix(alpha, beta, gamma)

    # positions of rotated points for each representation
    x_new_e = rotate_euler(alpha, beta, gamma, x)
    x_new_r = R.dot(x)
    x_new_aa = rodrigues_formula(n, x, theta)

    #Ground Truth Checking
    q_gt = Rot.from_euler('ZYZ', [alpha, beta, gamma]).as_quat() #Note: scipy defines this as extrinsic rotation.
    R_gt = Rot.from_euler('ZYZ', [alpha, beta, gamma])
    n_temp = Rot.from_quat(q_gt).as_rotvec()
    theta_gt = np.linalg.norm(n_temp)
    n_gt = n_temp / theta_gt 
    x_new_gt = R_gt.apply(x)

    print('-'*20)
    print('Euler angles:', np.degrees(alpha), np.degrees(beta), np.degrees(gamma))
    print('Axis angle:', n, np.degrees(theta))
    print('Ground Truth Axis Angle:', n_gt, np.degrees(theta_gt))
    print('Rotation matrix:', R)
    print('Ground Truth Rotation matrix:', R_gt)
    print('x_new_e:', x_new_e)
    print('Ground Truth x_new_e:', x_new_gt)
    print('x_new_r:', x_new_r)
    print('Ground Truth x_new_r:', x_new_gt)
    print('x_new_aa:', x_new_aa)
    print('Ground Truth x_new_aa:', x_new_gt)
    print('-'*20)

    point_e.set_base_pos_orient(x_new_e)
    point_r.set_base_pos_orient(x_new_r)
    point_aa.set_base_pos_orient(x_new_aa)

    # NOTE: Press enter to continue to next angles
    print('Press enter in the simulator to continue to the next angle set')
    keys = m.get_keys()
    while True:
        keys = m.get_keys()
        if 'return' in keys:
            break
        m.step_simulation(realtime=True)
    m.step_simulation(steps=50, realtime=True)
