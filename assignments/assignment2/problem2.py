from scipy.interpolate import CubicSpline
import os
import numpy as np
import mengine as m
from scipy.linalg import expm
np.set_printoptions(precision=3, suppress=True)

# NOTE: This assignment asks you to Implement FK using screw coordinates.
# Create environment and ground plane
env = m.Env()

def _hat(w):
    """Skew-symmetric matrix [w]^ from a 3-vector."""
    wx, wy, wz = w
    return np.array([[0,   -wz,  wy],
                     [wz,   0,  -wx],
                     [-wy,  wx,  0]], dtype=float)
def reset_sim():
    env.reset()
    ground = m.Ground([0, 0, -0.5])
    env.set_gui_camera(look_at_pos=[0, 0, 0.5], distance=1.5)
    # Create example robot
    robot = m.URDF(filename=os.path.join(
        m.directory, 'assignments', 'example_arm.urdf'), static=True, position=[0, 0, 0])
    robot.controllable_joints = [0, 1, 2]
    robot.end_effector = 3
    robot.update_joint_limits()
    return robot


def sample_configuration():
    # Sample a random configuration for the robot.
    # NOTE: Be conscious of joint angle limits
    # output: q: joint angles of the robot
    return np.random.uniform(low=[-np.pi, -np.pi/2, -np.pi/2], high=[np.pi/2, np.pi/2, np.pi/2])


def get_exp_coordinates(omega, v, theta):
    # NOTE It can be helpful to implement and use this function,
    # but it is not required. You can also perform all calculations in calculate_FK.
    # You should use expm(w*theta) to compute the matrix exponential
    #
    # Calculate the exponential coordinates (exp([S]theta)) of a screw
    # input: omega: angular part
    #        v: linear part
    #        theta: angle of rotation
    # output: E: exponential coordinates of the screw (4x4 matrix)
    # ------ TODO Student answer below -------
    eps = 1e-12
    omega = np.asarray(omega, dtype=float).reshape(3)
    v     = np.asarray(v,     dtype=float).reshape(3)

    w_norm = np.linalg.norm(omega)

    # Pure translation case (||omega|| ~ 0): exp([S]θ) = [[I, v θ], [0, 1]]
    if w_norm < eps:
        E = np.eye(4)
        E[:3, :3] = np.eye(3)
        E[:3, 3]  = v * theta
        return E

    # Ensure unit screw axis for the closed-form; rescale (S, θ) consistently:
    # exp([S]θ) = exp([Ŝ] θ_eff), where Ŝ = (omega_hat, v_hat), θ_eff = θ * ||omega||
    omega_hat = omega / w_norm
    v_hat     = v / w_norm
    theta_eff = theta * w_norm

    W = _hat(omega_hat)

    # Rotation block: e^{[ω] θ}
    R = expm(W * theta_eff)

    # V(θ) = I θ + (1 - cosθ)[ω] + (θ - sinθ)[ω]^2
    I3 = np.eye(3)
    W2 = W @ W
    Vtheta = I3 * theta_eff + (1.0 - np.cos(theta_eff)) * W + (theta_eff - np.sin(theta_eff)) * W2

    p = Vtheta @ v_hat

    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3]  = p
    return E

def rot_to_quat_xyzw(R):
    """Convert 3x3 rotation to quaternion [x, y, z, w]."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        S  = np.sqrt(tr + 1.0) * 2.0
        w  = 0.25 * S
        x  = (R[2,1] - R[1,2]) / S
        y  = (R[0,2] - R[2,0]) / S
        z  = (R[1,0] - R[0,1]) / S
    else:
        # Find the largest diagonal
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            S  = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w  = (R[2,1] - R[1,2]) / S
            x  = 0.25 * S
            y  = (R[0,1] + R[1,0]) / S
            z  = (R[0,2] + R[2,0]) / S
        elif i == 1:
            S  = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w  = (R[0,2] - R[2,0]) / S
            x  = (R[0,1] + R[1,0]) / S
            y  = 0.25 * S
            z  = (R[1,2] + R[2,1]) / S
        else:
            S  = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w  = (R[1,0] - R[0,1]) / S
            x  = (R[0,2] + R[2,0]) / S
            y  = (R[1,2] + R[2,1]) / S
            z  = 0.25 * S
    q = np.array([x, y, z, w], dtype=float)
    # normalize to be safe
    q /= np.linalg.norm(q)
    return q
    

def calculate_FK(q, joint=3):
    # Calculate the forward kinematics of the robot
    # NOTE: Use screw coordinate representation and the product of exponentials formulation
    # You should use expm(w*theta) to compute the matrix exponential
    # input: q: joint angles of the robot
    #        joint: index of the joint to calculate the FK for. 0 is the base joint, and 3 is the end effector
    # output: ee_position: position of the end effector
    #         ee_orientation: orientation of the end effector as a quaternion
    # ------ TODO Student answer below -------
    # orientation = m.get_quaternion(orientation) # NOTE: If you used transformation matrices, call this function to get a quaternion

    l1, l2, l3 = 0.5, 0.4, 0.3

    S_OMEGAS = [
        np.array([0.0, 0.0, 1.0]),  # ω1
        np.array([1.0, 0.0, 0.0]),  # ω2
        np.array([1.0, 0.0, 0.0]),  # ω3
    ]

    # Linear parts v = -ω × q  with q on each axis at home
    S_VS = [
        np.array([0.0, 0.0, 0.0]),   # v1 (q1 = [0,0,0])
        np.array([0.0, l1, 0.0]),    # v2 (q2 = [0,0,l1])
        np.array([0.0, l1+l2, 0.0]), # v3 (q3 = [0,0,l1+l2])
    ]

    # Home pose(s): at q=0 the EE sits at z = l1+l2+l3, with identity orientation
    M_ee = np.eye(4)
    M_ee[:3, 3] = [0.0, 0.0, l1 + l2 + l3]  # [0, 0, 1.2]

    # If your FK routine expects a list indexed by 'joint',
    # you can keep identities for intermediate frames and M_ee for index 3.
    M_LIST = [
        np.eye(4),  # (optional) frame at joint 0
        np.eye(4),  # (optional) frame at joint 1
        np.eye(4),  # (optional) frame at joint 2
        M_ee,       # end-effector home pose
    ]
    q = np.asarray(q, float).flatten()
    N = min(len(S_OMEGAS), len(S_VS))
    assert len(M_LIST) >= joint + 1, "M_LIST must contain the home transform for the requested frame."

    # Build transform up to the requested joint
    T = np.eye(4)
    for i in range(min(joint, N)):
        T = T @ get_exp_coordinates(S_OMEGAS[i], S_VS[i], q[i])

    # Multiply by the chosen home transform (frame 'joint')
    T = T @ M_LIST[joint]

    p = T[:3, 3]
    R = T[:3, :3]
    q_xyzw = rot_to_quat_xyzw(R)  # your convention: [x, y, z, w]
    return p, q_xyzw
    # ------ Student answer above -------


def compare_FK(ee_positions, ee_positions_pb, ee_orientations, ee_orientations_pb):
    # Compare the FK implementation to the built-in one
    # input: ee_positions: list of positions of the end effector
    #        ee_positions_pb: list of positions of the end effector from pybullet
    #        ee_orientations: list of orientations of the end effector
    #        ee_orientations_pb: list of orientations of the end effector from pybullet
    distance_error_sum = 0
    orientation_error_sum = 0
    for p1, p2 in zip(ee_positions, ee_positions_pb):
        distance_error_sum += np.linalg.norm(p1 - p2)
    for q1, q2 in zip(ee_orientations, ee_orientations_pb):
        error = np.arccos(2*np.square(q1.dot(q2)) - 1)
        orientation_error_sum += 0 if np.isnan(error) else error
    print('Average FK distance error:', distance_error_sum / len(ee_positions))
    print('Average FK orientation error:',
          orientation_error_sum / len(ee_orientations))


# ##########################################
# Problem 2:
# Forward Kinematics using screw coordinates
# ##########################################
robot = reset_sim()

# test cases
q_test = np.array([[0, 0, 0], [-0.3, 0.7, 0.9], [0.8, 1.4, 1.2]])
for q_i, idx in zip(q_test, range(3)):
    ee_pos, ee_orient = calculate_FK(q_i, joint=3)
    print("ee position and orientation for testcase ",
          idx, ": ", ee_pos, ee_orient)

ee_positions = []
ee_orientations = []

ee_positions_pb = []
ee_orientations_pb = []

for i in range(1000):
    # sample a random configuration q
    q = sample_configuration()
    # move robot into configuration q
    robot.control(q, set_instantly=True)
    m.step_simulation(realtime=True)
    # calculate ee_position, ee_orientation using calculate_FK
    ee_position, ee_orientation = calculate_FK(q, joint=3)

    ee_positions.append(ee_position)
    ee_orientations.append(ee_orientation)
    # calculate ee position, orientation using pybullet's FK
    ee_position_pb, ee_orientation_pb = robot.get_link_pos_orient(
        robot.end_effector)
    ee_positions_pb.append(ee_position_pb)

    ee_orientations_pb.append(ee_orientation_pb)
    # print(ee_position, ee_position_pb, ee_orientation, ee_orientation_pb)
    # m.Shape(m.Sphere(radius=0.02), static=True, position=ee_position, collision=False, rgba=[1, 0, 0, 1])
    # m.Shape(m.Sphere(radius=0.02), static=True, position=ee_position_pb, collision=False, rgba=[0, 1, 0, 1])

# compare your implementation and pybullet's FK
compare_FK(ee_positions, ee_positions_pb, ee_orientations, ee_orientations_pb)

# ##########################################
