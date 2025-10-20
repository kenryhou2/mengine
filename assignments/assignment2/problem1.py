import numpy as np
import os
import math
import mengine as m

"""
16-741 Assignment 2 Problem 1.

Attention: quaternions are represented as [x, y, z, w], same as in pybullet.

"""

np.set_printoptions(precision=4, suppress=True)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix R (3x3) to quaternion q (1x4) xyzw"""
    # input: R: rotation matrix
    # output: q: quaternion
    # ------ TODO: Student answer below -------
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

   # ------ Student answer above -------


def rodrigues_formula(n, x, theta):
    # Rodrigues' formula for axis-angle: rotate a point x around an axis n by angle theta
    # input: n, x, theta: axis, point, angle
    # output: x_new: new point after rotation
    # ------ TODO Student answer below -------
    n = n / np.linalg.norm(n)
    x_rot = x * np.cos(theta) + np.cross(n, x) * np.sin(theta) + n * (np.dot(n, x)) * (1 - np.cos(theta))
    return x_rot
    # ------ Student answer above -------


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    x, y, z, w = q
    return np.array([-x, -y, -z, w], float)

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n

def rotate_point_quat(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate 3D point x by unit quaternion q (xyzw). By performing q * x * q^-1."""
    q = quat_normalize(q)
    v = np.array([x[0], x[1], x[2], 0.0], float)
    v_rot = hamilton_product(hamilton_product(q, v), quat_conjugate(q))
    return v_rot[:3]



def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle representation to quaternion."""
    # input: axis: axis of rotation
    #        angle: angle of rotation (radians)
    # output: q: quaternion
    # ------ TODO: Student answer below -------
    
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a = axis / n
    half = 0.5 * angle
    s = np.sin(half)
    q = np.array([a[0]*s, a[1]*s, a[2]*s, np.cos(half)], float)
    return quat_normalize(q)
    # ------ Student answer above -------


def hamilton_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    # ------ TODO: Student answer below -------
    p = np.asarray(p, dtype=float).reshape(4)
    q = np.asarray(q, dtype=float).reshape(4)

    px, py, pz, pw = p
    qx, qy, qz, qw = q

    vp = np.array([px, py, pz])
    vq = np.array([qx, qy, qz])

    # vector part:  pw*vq + qw*vp + vp × vq
    v = pw * vq + qw * vp + np.cross(vp, vq)
    # scalar part:  pw*qw - vp · vq
    s = pw * qw - np.dot(vp, vq)

    return np.array([v[0], v[1], v[2], s])
    # ------ Student answer above -------


def unit_tests():
    """Simple unit tests.
    Passing these test cases does NOT ensure your implementation is fully correct.
    """
    # test rotation_matrix_to_quaternion
    q = rotation_matrix_to_quaternion(np.diag([1, -1, -1]))
    try:
        assert np.allclose(q, [1, 0, 0, 0]) or np.allclose(q, [-1, 0, 0, 0])
        print("rotation_matrix_to_quaternion passed test case 1")
    except AssertionError:
        print("rotation_matrix_to_quaternion failed test case 1")

    R = np.array([[-0.545, 0.797, 0.260],
                  [0.733, 0.603, -0.313],
                  [-0.407, 0.021, -0.913]])
    q = rotation_matrix_to_quaternion(R)
    try:
        assert np.allclose(q, [0.437, 0.875, -0.0836, 0.191], atol=1e-3)
        print("rotation_matrix_to_quaternion passed test case 2")
    except AssertionError:
        print("rotation_matrix_to_quaternion failed test case 2")

    # test axis_angle_to_quaternion
    q = axis_angle_to_quaternion(np.array([1, 0, 0]), 0.123)
    try:
        assert np.allclose(q, [0.06146124, 0, 0, 0.99810947])
        print("axis_angle_to_quaternion passed test case")
    except AssertionError:
        print("axis_angle_to_quaternion failed test case")

    # test hamilton_product
    p = np.array([0.437, 0.875, -0.0836, 0.191])
    q = np.array([0.06146124, 0, 0, 0.99810947])
    try:
        assert np.allclose(hamilton_product(p, q),
                           [0.4479,  0.8682, -0.1372,  0.1638], atol=1e-3)
        print("hamilton_product passed test case")
    except AssertionError:
        print("hamilton_product failed test case")


if __name__ == '__main__':
    # Create environment and ground plane
    env = m.Env()
    ground = m.Ground([0, 0, -0.5])
    env.set_gui_camera(look_at_pos=[0, 0, 0])

    # Axis-angle definition
    n = np.array([0, 0, 1])
    x = np.array([0.2, 0, 0])

    # Create axis
    axis = m.Shape(m.Cylinder(radius=0.02, length=0.5), static=True, position=[0, 0, 0], orientation=n,
                   rgba=[0.8, 0.8, 0.8, 1])
    # Create point to rotate around axis
    point = m.Shape(m.Sphere(radius=0.02), static=True,
                    position=x, rgba=[0, 0, 1, 0.5])
    point_q = m.Shape(m.Sphere(radius=0.02), static=True,
                      position=x, rgba=[0, 1, 0, 0.5])

    # First we want to implement some converstions and the Hamilton product for quaternions.
    print("Running unit tests...")
    unit_tests()

    x_new_report = []
    x_new_q_report = []

    for i in range(10000):
        theta = np.radians(i)
        # Rodrigues' formula for axis-angle rotation
        x_new = rodrigues_formula(n, x, theta)

        # Axis-angle to quaternion
        theta = np.radians(i-10)  # Offset theta so we can see the two points
        q = axis_angle_to_quaternion(n, theta)

        # rotate using quaternion and the hamilton product
        # ------ TODO Student answer below -------
        x_new_q = np.zeros(3)
        x_new_q = rotate_point_quat(x, q)
        # ------ Student answer above -------

        point.set_base_pos_orient(x_new)
        point_q.set_base_pos_orient(x_new_q[:3])

        m.step_simulation(realtime=True)

        if i % 50 == 0 and i < 501:
            x_new_report.append(x_new.tolist())
            x_new_q_report.append(x_new_q[:3].tolist())
        if i == 500:
            print("Point rotated using rodrigues formula: ")
            for row in x_new_report:
                formatted_row = [f"{elem:.4f}" for elem in row]
                print(formatted_row)
            print("Point rotated using hamilton product: ")
            for row in x_new_q_report:
                formatted_row = [f"{elem:.4f}" for elem in row]
                print(formatted_row)
