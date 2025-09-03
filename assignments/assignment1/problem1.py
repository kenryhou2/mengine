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

    def qmul_xyzw(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])


    def q_from_axis_angle_xyzw(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-16)
        half = theta * 0.5
        s = np.sin(half)
        x, y, z = axis * s
        w = np.cos(half)
        return np.array([x, y, z, w])

    def eulerZYZ_to_quat_xyzw(alpha, beta, gamma):
        qz_a = q_from_axis_angle_xyzw([0,0,1], alpha)
        qy_b = q_from_axis_angle_xyzw([0,1,0], beta)
        qz_g = q_from_axis_angle_xyzw([0,0,1], gamma)
        q = qmul_xyzw(qmul_xyzw(qz_a, qy_b), qz_g)
        return q / np.linalg.norm(q)


    def quat_xyzw_to_axis_angle(q, eps=1e-12):
        """
        Convert a quaternion in [x, y, z, w] to (axis, angle).
        - q can be non-normalized; we normalize internally.
        - Returns axis as a unit 3-vector, angle in radians ∈ [0, π].
        """
        q = np.asarray(q, dtype=float)
        # normalize defensively
        n = np.linalg.norm(q)
        if n < eps:
            # invalid quaternion; return identity
            return np.array([1.0, 0.0, 0.0]), 0.0
        x, y, z, w = q / n

        # clamp for numerical safety
        w = np.clip(w, -1.0, 1.0)
        theta = 2.0 * np.arccos(w)             # angle ∈ [0, π]

        s = np.sqrt(max(0.0, 1.0 - w*w))       # == sin(theta/2)
        if s < eps:
            # angle ~ 0: axis is undefined; pick a default
            return np.array([1.0, 0.0, 0.0]), 0.0

        axis = np.array([x, y, z]) / s
        # ensure unit axis (defensive)
        axis = axis / (np.linalg.norm(axis) + eps)
        return axis, theta
    
    axis, theta = quat_xyzw_to_axis_angle(eulerZYZ_to_quat_xyzw(alpha, beta, gamma))
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
    q_gt = Rot.from_euler('ZYZ', [alpha, beta, gamma]).as_quat()
    R_gt = Rot.from_euler('ZYZ', [alpha, beta, gamma])
    n_temp = Rot.from_quat(q_gt).as_rotvec()
    theta_gt = np.linalg.norm(n_temp)
    n_gt = n_temp / theta_gt if theta_gt > 1e-12 else np.array([1,0,0])
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
