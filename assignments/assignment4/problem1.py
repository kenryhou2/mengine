import os
import numpy as np
import mengine as m
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError

env = m.Env(gravity=[0, 0, 0])


def moveto(robot, robot_marker, pos):
    # moves robot and robot reference frame ('robot_marker') to position pos
    robot.set_base_pos_orient(
        pos+np.array([0.1, -0.1, 0.]), m.get_quaternion([np.radians(90), 0, 0]))
    robot_marker.set_base_pos_orient(
        pos, [0, 0, 0, 1])
    
def minkowski_cspace_polygon(rect, neg_robot):
    pts = (rect[None,:,:] + neg_robot[:,None,:]).reshape(-1,2)
    pts = np.unique(pts, axis=0)
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def reset():
    # Create environment and ground plane
    env.reset()
    ground = m.Ground(position=[0, 0, -0.02])
    env.set_gui_camera(look_at_pos=[0.5, 0.5, 0], distance=0.7, pitch=-89.99)

    robot_init_position = np.array([0.0, 0, 0.0])
    robot = m.Shape(m.Mesh(filename=os.path.join(m.directory, 'triangle.obj'), scale=[
        1, 0.1, 1]), static=False, position=robot_init_position, orientation=m.get_quaternion([np.radians(90), 0, 0]), rgba=[0, 1, 0, 0.5])
    # mark robot reference frame
    robot_marker = m.Shape(m.Sphere(radius=0.02), static=False, collision=False,
                           position=robot_init_position+np.array([-0.1, 0.1, 0.]), rgba=[1, 1, 1, 1])
    l1 = 0.3; h1 = 0.48; l2 = 0.4; h2 = 0.36
    c1 = [0.5, 0.5]; c2 = (0.9, 0.75)
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)

    obstacle1 = m.Shape(m.Box(half_extents=[l1/2, h1/2, 0.01]), static=True, position=[
                        c1[0], c1[1], 0.0], rgba=[1, 1, 0, 1])
    obstacle2 = m.Shape(m.Box(half_extents=[l2/2, h2/2, 0.01]), static=True, position=[
                        c2[0], c2[1], 0.0], rgba=[1, 1, 0, 1])

    m.step_simulation(realtime=True)
    print("Computing Minkowski difference...")

# ------ TODO Student answer below -------
# Compute C-space obstacle for robot and obstacles using Minkowski sum with the
# negated robot (C-obstacle = Obstacle ⊕ (−Robot))

    # 1) Define the rectangle (all 4 corners) from center c1 and size l1 x h1
    rect1 = np.array([
        [c1[0] - l1/2, c1[1] - h1/2],  # bottom-left
        [c1[0] + l1/2, c1[1] - h1/2],  # bottom-right
        [c1[0] + l1/2, c1[1] + h1/2],  # top-right
        [c1[0] - l1/2, c1[1] + h1/2],  # top-left
    ], dtype=float)

    rect2 = np.array([
    [c2[0] - l2/2, c2[1] - h2/2],
    [c2[0] + l2/2, c2[1] - h2/2],
    [c2[0] + l2/2, c2[1] + h2/2],
    [c2[0] - l2/2, c2[1] + h2/2],
    ], dtype=float)

    # 2) Robot triangle in 2D (your footprint in the plane)
    triangle = np.array([
        [-0.1,  0.1],
        [ 0.1,  0.1],
        [ 0.0, -0.1],
    ], dtype=float)

    # 3) Reflect the robot for C-obstacle construction
    neg_robot = -triangle  # reflect about origin

    cobs1 = minkowski_cspace_polygon(rect1, neg_robot)
    cobs2 = minkowski_cspace_polygon(rect2, neg_robot)

    plt.figure()
    plt.fill(cobs1[:,0], cobs1[:,1], alpha=0.5, color='red', label='C-space obstacle 1')
    plt.fill(cobs2[:,0], cobs2[:,1], alpha=0.5, color='orange', label='C-space obstacle 2')

    plt.fill(rect1[:,0], rect1[:,1], alpha=0.3, color='green', label='Obstacle 1 (workspace)')
    plt.fill(rect2[:,0], rect2[:,1], alpha=0.3, color='yellow', label='Obstacle 2 (workspace)')

    plt.fill(triangle[:,0], triangle[:,1], alpha=0.4, color='blue', label='Robot shape')

    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("C-space obstacles = O₁ ⊕ (−R) and O₂ ⊕ (−R)")
    plt.show()


# ------ Student answer above -------


reset()
