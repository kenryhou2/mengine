import time
import os
import mengine as m
import numpy as np


def invertQ(q):
    """
    Invert a quaternion, this function is optional and you could use it in line_intersection if you want
    """
    # ------ TODO Student answer below -------
    # NOTE: Optional, you do not need to use this function
    return np.array([0, 0, 0, 1])
    # ------ Student answer above -------


def line_intersection(p1, p2, q1, q2, tol=1e-3):
    """
    Intersection of:
      - mode="segment": two line segments p1->p2 and q1->q2
      - mode="line": the infinite lines through those segments

    Returns:
      - np.array shape (3,) for a single intersection point (when lines meet)
      - None if lines are parallel (and not colinear) or skew (no single intersection)
      - For colinear overlaps, returns None (simple behavior; can be extended).
    """
    p1, p2, q1, q2 = map(lambda x: np.asarray(x, float), (p1, p2, q1, q2))
    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1

    a = np.dot(u, u)      # ||u||^2
    b = np.dot(u, v)      # u·v
    c = np.dot(v, v)      # ||v||^2
    d = np.dot(u, w0)     # u·w0
    e = np.dot(v, w0)     # v·w0

    denom = a*c - b*b

    # Parallel (or nearly)
    if abs(denom) < tol:
        # Optional: detect colinearity; simple exit for now.
        # If you want to treat colinear overlaps specially, add logic here.
        print('Lines are parallel or nearly parallel')
        return None

    # Solve for parameters on the infinite lines
    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom

    # Points on each line
    P = p1 + s*u
    Q = q1 + t*v

    # Lines intersect iff P ≈ Q (otherwise skew)
    if np.linalg.norm(P - Q) >= tol:
        print('Lines are skew and do not intersect')
        return None  # skew: closest points are distinct

    # Return the (averaged) intersection point
    return 0.5 * (P + Q)


def test_line_intersection():
    p1=[-1, 0, 0]
    p2=[1, 0, 0]
    q1=[0, -1, 0]
    q2=[0, 1, 0]
    intersection=line_intersection(p1, p2, q1, q2)
    print('Intersection should be [0,0,0]: ', intersection)
    p1=[0, 0, 0]
    p2=[1, 1, 1]
    q1=[1, 0, 0]
    q2=[1, 0, 0]
    q2=[0, 1, 1]
    intersection=line_intersection(p1, p2, q1, q2)
    print('Intersection should be [0.5, 0.5, 0.5]: ', intersection)
    p1=[0, 0, 0]
    p2=[1, 0, 0]
    q1=[0, 0, 1]
    q2=[0, 1, 1]
    intersection=line_intersection(p1, p2, q1, q2)
    print('Intersection should be None: ', intersection)
    p1=[0, 0, 0]
    p2=[1, 0, 0]
    q1=[2, -1, 0]
    q2=[2, 1, 0]
    intersection=line_intersection(p1, p2, q1, q2)
    print('Intersection should be [2, 0, 0]: ', intersection)

# test_line_intersection()

# Create environment and ground plane
env = m.Env()
# ground = m.Ground()
env.set_gui_camera(look_at_pos=[0, 0.4, 0.25])

fbl = m.URDF(filename=os.path.join(m.directory, 'fourbarlinkage.urdf'),
             static=True, position=[0, 0, 0.3], orientation=[0, 0, 0, 1])
fbl.controllable_joints = [0, 1, 2]
# Create a constraint for the 4th joint to create a closed loop
fbl.create_constraint(parent_link=1, child=fbl, child_link=4, joint_type=m.p.JOINT_POINT2POINT, joint_axis=[
                      0, 0, 0], parent_pos=[0, 0, 0], child_pos=[0, 0, 0])
m.step_simulation(steps=20, realtime=False)

coupler_links = [1, 3, 5]

links = [1, 3]
global_points = []
previous_global_points = []
lines = [None, None]
lines_start_end = [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]

for link in links:
    global_points.append(fbl.get_link_pos_orient(link)[0])
    previous_global_points.append(global_points[-1])
    point = m.Shape(m.Sphere(radius=0.02), static=True,
                    position=global_points[-1], rgba=[0, 0, 1, 1])

intersect_points_local = []
intersect_points_local_bodies = []

for i in range(10000):
    fbl.control([np.radians(i)]*3)

    if i > 3:
        for j, (link, global_position, previous_global_position) in enumerate(zip(links, global_points, previous_global_points)):
            p_new = fbl.get_link_pos_orient(link)[0]
            ic_vector_of_motion = p_new - previous_global_position
            ic_bisector = np.cross(ic_vector_of_motion, [0, 1, 0])
            ic_bisector = ic_bisector / np.linalg.norm(ic_bisector)
            previous_global_points[j] = p_new

            lines[j] = m.Line(p_new-ic_bisector, p_new+ic_bisector,
                              radius=0.005, rgba=[0, 0, 1, 0.5], replace_line=lines[j])
            lines_start_end[j] = (p_new-ic_bisector, p_new+ic_bisector)

        if len(intersect_points_local) < 400:
            # stop drawing if we have drawn 500 points
            intersect_point = line_intersection(
                lines_start_end[0][0], lines_start_end[0][1], lines_start_end[1][0], lines_start_end[1][1])
            print('Intersection point: ', intersect_point)
            if intersect_point is not None:
                m.Shape(m.Sphere(radius=0.005), static=True,
                        position=intersect_point, collision=False, rgba=[1, 0, 0, 1])
                # draw moving centrode
                # get intersection point in local frame w.r.t. link 4
                p, _ = fbl.global_to_local_coordinate_frame(intersect_point, link=3)
                local_intersect_point = np.array(p)

                intersect_points_local.append(local_intersect_point)
                # get global coordinates of intersection point
                intersect_point_local_body = m.Shape(m.Sphere(radius=0.005), static=True,
                                                     position=intersect_point, collision=False, rgba=[0, 1, 0, 1])
                intersect_points_local_bodies.append(
                    intersect_point_local_body)

        # redraw intersection points of moving centrode
        for body, point_local in zip(intersect_points_local_bodies, intersect_points_local):
            p, _ = fbl.local_to_global_coordinate_frame(point_local, link=3)
            body.set_base_pos_orient(p)

    m.step_simulation(realtime=True)

    if i == 500 or i == 600 or i == 700:
        print('--------------------------------------------------------------')
        print(f'Frame {i}: Please save screenshot and include in writeup')
        input("Press Enter to continue...")
