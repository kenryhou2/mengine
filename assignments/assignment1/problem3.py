from scipy.interpolate import CubicSpline
import os
import numpy as np
import mengine as m
import pybullet as p

# ---------- helpers ----------
def quat_xyzw_to_R(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def R_from_axis_angle(axis, theta):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    ax, ay, az = a
    K = np.array([[0, -az, ay],
                  [az, 0, -ax],
                  [-ay, ax, 0]], dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return np.eye(3) + s*K + (1 - c)*(K @ K)

def _T_from_pos_quat(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_R(quat_xyzw)
    T[:3, 3]  = np.asarray(pos, float)
    return T

def _T_from_axis_angle(axis, theta):
    T = np.eye(4)
    T[:3, :3] = R_from_axis_angle(axis, theta)
    return T

def rotmat_to_quat_xyzw(R):
    """3x3 rotation matrix -> quaternion [x,y,z,w]."""
    R = np.asarray(R, dtype=float)
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        # find the largest diagonal
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= (np.linalg.norm(q) + 1e-16)
    return q

def get_joint_frame(robot, joint_index):
    """
    Returns:
      parent_pos (3,), parent_quat (4,),
      axis (3,),
      child_pos (3,), child_quat (4,)
    All frames are URDF joint frames relative to parent/child links.
    """
    ji = p.getJointInfo(robot.body, joint_index, physicsClientId=robot.id)
    axis         = np.array(ji[13])
    parent_pos   = np.array(ji[14])
    parent_quat  = np.array(ji[15])
    # Some builds include child frame at indices 18/19
    if len(ji) > 19:
        child_pos  = np.array(ji[18])
        child_quat = np.array(ji[19])
    else:
        child_pos  = np.zeros(3)
        child_quat = np.array([0,0,0,1], dtype=float)
    return parent_pos, parent_quat, axis, child_pos, child_quat
# ---------- end helpers ----------

np.set_printoptions(precision=3, suppress=True) 

# NOTE: This assignment asks you to Implement FK, plot the robot workspace, and check for collisions
# Create environment and ground plane
env = m.Env()


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
    # ------ TODO Student answer below -------
    # print('TODO sample_configuration')
    # return np.zeros(3)
    q0_limits = [-np.pi, np.pi/2]
    q1_limits = [-np.pi/2, np.pi/2]
    q2_limits = [-np.pi/2, np.pi/2]
    q0 = np.random.uniform(q0_limits[0], q0_limits[1])
    q1 = np.random.uniform(q1_limits[0], q1_limits[1])
    q2 = np.random.uniform(q2_limits[0], q2_limits[1])
    return np.array([q0, q1, q2])

    # ------ Student answer above -------


def calculate_FK(q, joint=3):
    # Calculate the forward kinematics of the robot
    # NOTE: We encourage doing this with transformation matrices, as shown in class
    # input: q: joint angles of the robot
    #        joint: index of the joint to calculate the FK for
    # output: ee_position: position of the end effector
    #         ee_orientation: orientation of the end effector
    # ------ TODO Student answer below -------
    # print('TODO calculate_FK')
    position = np.zeros(3)
    orientation = np.zeros(4)
    # orientation = m.get_quaternion(orientation) # NOTE: If you used transformation matrices, call this function to get a quaternion
    
    #Obtain World to base transform.
    # World -> base
    base_pos, base_quat = p.getBasePositionAndOrientation(robot.body, physicsClientId=robot.id)
    T = _T_from_pos_quat(base_pos, base_quat)



    # Chain joints 0..joint-1
    for i in range(joint):
        parent_pos, parent_quat, axis, child_pos, child_quat = get_joint_frame(robot, i)
        T_parent_to_joint = _T_from_pos_quat(parent_pos, parent_quat)
        T_joint_motion    = _T_from_axis_angle(axis, float(q[i]))
        T_joint_to_child  = _T_from_pos_quat(child_pos, child_quat)
        
        #adding link length for end effector
        if i == 2 and joint == 3:
            T_joint_to_child[:3, 3]  = np.array([0, 0, 0.3])
        if i == 1 and joint == 2:
            T_joint_to_child[:3, 3]  = np.array([0, 0, 0.4])
        if i == 0 and joint == 1:
            T_joint_to_child[:3, 3]  = np.array([0, 0, 0.5])  

        # print(f"Joint {i}:")
        # print(" T_parent_to_joint:\n", T_parent_to_joint)
        # print(" T_joint_motion:\n", T_joint_motion)
        # print(" T_joint_to_child:\n", T_joint_to_child)
        
        # world -> ... -> parent * (parent->joint) * (motion) * (joint->child link)
        T = T @ T_parent_to_joint @ T_joint_motion @ T_joint_to_child

    position = T[:3, 3].copy()
    orientation = rotmat_to_quat_xyzw(T[:3,:3])  # returns [x,y,z,w]

    # ------ Student answer above -------
    return position, orientation


def compare_FK(ee_positions, ee_positions_pb, ee_orientations, ee_orientations_pb):
    # Compare the FK implementation to the built-in one
    # input: ee_positions: list of positions of the end effector
    #        ee_positions_pb: list of positions of the end effector from pybullet
    #        ee_orientations: list of orientations of the end effector (normalized quaternions)
    #        ee_orientations_pb: list of orientations of the end effector from pybullet
    distance_error_sum = 0
    orientation_error_sum = 0
    for p1, p2 in zip(ee_positions, ee_positions_pb):
        distance_error_sum += np.linalg.norm(p1 - p2)
    for q1, q2 in zip(ee_orientations, ee_orientations_pb):
        error = np.arccos(2*np.square(q1.dot(q2)) - 1)
        orientation_error_sum += 0 if np.isnan(error) else error
    print('Average FK distance error:', distance_error_sum / len(ee_positions))
    print('Average FK orientation error:', orientation_error_sum / len(ee_orientations))


def plot_point(position):
    # input: position: list of [x,y,z] position of the end effector
    m.Shape(m.Sphere(radius=0.01), static=True, position=position, collision=False, rgba=[1, 0, 0, 1])


def wait_for_enter():
    # NOTE: Press enter to continue to next problem
    print('Press enter in the simulator to continue to the next problem')
    keys = m.get_keys()
    while True:
        keys = m.get_keys()
        if 'return' in keys:
            break
        m.step_simulation(realtime=True)
    m.step_simulation(steps=50, realtime=True)


def check_collision(q, box_position, box_half_extents):
    # Check if the robot is in collision region
    # input: q: joint angles of the robot
    #        box_position: position of the collision region
    #        box_half_extents: half extents of the collision region
    # output: in_collision: True if the robot is in collision region, False otherwise
    # ------ TODO Student answer below -------
    # print('TODO check_collision')
    # return False

    collision_status = False
    num_joints = len(q)
    #Check collision for each link
    for joint_index in range(num_joints):
        link_pos, link_orientation = calculate_FK(q, joint=joint_index)
        #Check if link_pos is within the box defined by box_position and box_half_extents
        if all(box_position - box_half_extents <= link_pos) and all(link_pos <= box_position + box_half_extents):
            collision_status = True
            break
    return collision_status

    # ------ Student answer above -------


# ##########################################
# Problem 3.1:
# Implement FK for a 3-link manipulator and compare your implementation to the built-in function in pybullet
# ##########################################
robot = reset_sim()
ee_positions = []
ee_orientations = []
ee_positions_pb = []
ee_orientations_pb = []

for i in range(100):
    if i % 10 == 0:
        print('Sampling configuration', i)
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
    ee_position_pb, ee_orientation_pb = robot.get_link_pos_orient(robot.end_effector)
    ee_positions_pb.append(ee_position_pb)
    ee_orientations_pb.append(ee_orientation_pb)
# compare your implementation and pybullet's FK
compare_FK(ee_positions, ee_positions_pb, ee_orientations, ee_orientations_pb)

# NOTE: Press enter to continue to problem 3.2
wait_for_enter()


# # ##########################################
# # Problem 3.2:
# # Plot the workspace of the robot using a sampling-based approach
# # ##########################################

# ------ TODO Student answer below -------
for i in range(1005):
    if i % 100 == 0:
        print('Sampling configuration', i)
    # sample a random configuration q
    # TODO
    q = sample_configuration()

    # move robot into configuration q
    robot.control(q, set_instantly=True)
    m.step_simulation(realtime=True)

    # calculate ee_position, ee_orientation using calculate_FK
    # TODO
    ee_position, ee_orientation = calculate_FK(q, joint=3)
    # plot workspace as points of the end effector
    plot_point(ee_position)
# ------ Student answer above -------

# NOTE: Press enter to continue to problem 3.3
wait_for_enter()


# ##########################################
# Problem 3.3:
# Expand FK to be FK to all joints and then check for collisions with a box region
# ##########################################

reset_sim()
# create a collision region
box_position = np.array([-0.3, 0, 1.0])
box_half_extents = np.array([0.15, 0.35, 0.25])

# Create a visual box to check collisions with
box = m.Shape(m.Box(box_half_extents), static=True, position=box_position, collision=False, rgba=[0, 1, 0, 0.5])

# Define a joint space trajectory for the robot to follow
q0 = np.array([-np.pi/4, -np.pi/2, -3*np.pi/4])
q1 = np.array([np.pi/2, np.pi/4, np.pi/2])
q2 = np.array([np.pi / 2, np.pi / 4, np.pi / 2])
t = np.array([1, 2, 3])
# Create a cubic spline interpolation function
q0_spline = CubicSpline(t, q0)
q1_spline = CubicSpline(t, q1)
q2_spline = CubicSpline(t, q2)
t_interp = np.linspace(t[0], t[-1], num=200)
q0_traj = q0_spline(t_interp)
q1_traj = q1_spline(t_interp)
q2_traj = q2_spline(t_interp)
traj = np.array([q0_traj, q1_traj, q2_traj])

t_collision = []
for i in range(200):
    # move robot to configuration
    target_joint_angles = traj[:, i]
    # print(target_joint_angles)
    robot.control(target_joint_angles, set_instantly=True)
    m.step_simulation(realtime=True)

    # check if robot is in collision region
    # TODO implement check_collision
    in_collision = check_collision(target_joint_angles, box_position, box_half_extents)
    if in_collision:
        box.change_visual(rgba=[1, 0, 0, 0.5])
        print('Robot is in collision region!')
        t_collision.append(i)
    else:
        box.change_visual(rgba=[0, 1, 0, 0.5])
print(t_collision)
m.step_simulation(steps=50, realtime=True)
