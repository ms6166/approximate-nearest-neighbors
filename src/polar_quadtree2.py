import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import random


def cartesian_to_polar(pts):
    """
    Args
        points: ndarray containing points in Cartesian coordinates.

    Returns
        ndarray: points in polar coordinates, where the first column is the
        radius and the second column is the angle (in radians).
    """
    assert pts.shape[1] == 2
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    angles[angles < 0] += 2.0 * np.pi  # Fix cases where arctan2 < 0
    return np.column_stack((np.linalg.norm(pts, axis=1), angles))


def polar_to_cartesian(pts):
    pt_x = pts[:, 0] * np.cos(pts[:, 1])
    pt_y = pts[:, 0] * np.sin(pts[:, 1])
    return np.column_stack((pt_x, pt_y))


def hyperbolic_distance(u, v):
    u = polar_to_cartesian(np.array(u))
    v = polar_to_cartesian(np.array(v))
    norm_uv = np.linalg.norm(u - v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    return np.arcosh(1 + 2 * (norm_uv ** 2 / ((1 - norm_u ** 2) * (1 - norm_v ** 2))))


def polar_equal_area_split(r_min, r_max, theta_min, theta_max, alpha=1.0):
    """
    Args
        r_min, r_max (float): polar radius range of current node
        theta_min, theta_max (float): angles (in degrees) range of the current node

    Returns
        (float, float): midpoint of radius and angle, split in half according to hyperbolic area.
    """
    r_mid = np.arccosh((np.cosh(alpha * r_max) + np.cosh(alpha * r_min)) / 2.0) * (1.0 / alpha)
    theta_mid = (theta_min + theta_max) / 2.
    return r_mid, theta_mid


def polar_equal_length_split(r_min, r_max, theta_min, theta_max):
    """
    Args
        r_min, r_max (float): polar radius range of current node
        theta_min, theta_max (float): angles (in degrees) range of the current node

    Returns
        (float, float): midpoint of radius and angle, split in half according to Euclidean length.
    """
    r_mid = (r_min + r_max) / 2.
    theta_mid = (theta_min + theta_max) / 2.
    return r_mid, theta_mid


class PolarNode:
    def __init__(self, r_min, r_max, theta_min, theta_max):
        """
        Creates a polar node for hyperbolic box with inner radius r_min, outer radius r_max
        and polar angle (in radians) range [min_theta, max_theta[
        """
        self.min_r = r_min
        self.max_r = r_max
        self.min_theta = theta_min
        self.max_theta = theta_max

        self.children = []
        self.points = []  # This is inefficient, should we store only indices or better not care about it for now?
        self.center_of_mass = None

    def max_radius(self):
        max_arc = hyperbolic_distance((self.max_r, self.min_theta), (self.max_r, self.max_theta))
        diagonal = hyperbolic_distance((self.min_r, self.min_theta), (self.max_r, self.max_theta))
        return max(max_arc, diagonal)

    def contains(self, test_point):
        """
        Args
            test_point (float, float): radius and polar angle of test point

        Returns
            True if PolarNode contains polar_point,
            False otherwise.
        """
        r, theta = test_point
        return self.min_r <= r <= self.max_r and self.min_theta <= theta <= self.max_theta

    def is_leaf(self):
        return len(self.points) == 1

    def is_empty(self):
        return len(self.points) == 0

    # Add new point to this particular node
    def append(self, point):
        # If the new point ISN'T in this box, the task is ill-posed
        if not self.contains(point):
            return False

        # If the new point IS in this box, but the box is empty, just add it in
        if self.is_empty():
            self.points.append((point[0], point[1]))
            self.update_center_of_mass()
            return True

        # If the new point IS in the this box, but the box is a NON-EMPTY LEAF, need to subdivide
        if self.is_leaf():
            # Add the point to leaf (not a leaf anymore)
            assert len(self.children) == 0
            self.points.append((point[0], point[1]))
            self.update_center_of_mass()

            # Compute splitting point (either by area or by length)
            mid_r, mid_theta = polar_equal_area_split(self.min_r, self.max_r, self.min_theta, self.max_theta)
            # mid_r, mid_theta = polar_equal_length_split(self.min_r, self.max_r, self.min_theta, self.max_theta)
            assert self.min_r < mid_r < self.max_r
            assert self.min_theta < mid_theta < self.max_theta

            # Generate four polar subnodes and add them as children to this one.
            # NOTE: Initially, we thought we only add in the non-empty subnodes, but this creates problems.
            #       So we sipmly add all of them.
            subnodes = [
                PolarNode(mid_r, self.max_r, self.min_theta, mid_theta),
                PolarNode(mid_r, self.max_r, mid_theta, self.max_theta),
                PolarNode(self.min_r, mid_r, mid_theta, self.max_theta),
                PolarNode(self.min_r, mid_r, self.min_theta, mid_theta)
            ]
            self.children.extend(subnodes)

            # Insert the new point + the (single) old leaf point into the correct subnodes
            for p in [point, self.points[0]]:
                for subnode in subnodes:
                    if subnode.append(p):
                        continue

            return True

        # Otherwise, it's NOT A LEAF and we need to recursively add to the correct children
        assert len(self.children) > 0
        self.points.append((point[0], point[1]))
        self.update_center_of_mass()
        for child in self.children:
            if child.append(point):  # if child accepts point (e.g. ITS children also done accepting), then we're done
                return True

        # For completeness. Won't reach here.
        return False

    def update_center_of_mass(self):
        # TODO: using Euclidean centroid (mean of the positions) as a placeholder
        # for the Einstein midpoint.
        cartesian_points = polar_to_cartesian(np.array(self.points))
        num = 0
        denom = 0
        cartesian_points = polar_to_cartesian(np.array(self.points))
        for point in cartesian_points:
            lorenz_factor = 1/(np.sqrt(1-(sum(pow(element, 2) for element in point)/(pow(c, 2)))))
            num += lorenz_factor*point
            denom += lorenz_factor
        self.center_of_mass = num/denom
        #self.center_of_mass = np.mean(cartesian_points, axis=0)

    def __str__(self):
        return f"Radius range: [{self.min_r}, {self.max_r}]; Angle range: [{self.min_theta}, {self.max_theta}]"

    # Render this node and recursively all its non-empty children too
    def render(self, ax, center=(0, 0)):
        wedge = patches.Wedge(center, r=self.max_r, theta1=np.degrees(self.min_theta),
                              theta2=np.degrees(self.max_theta), width=(self.max_r - self.min_r), color='cyan',
                              fill=False)
        ax.add_patch(wedge)

        # The rescursion. Important to note though that the non-displayed children still do exist in the hierarchy!
        for child in self.children:
            if not child.is_empty():
                child.render(ax, center)

    # Seems like a method for debugging / printing
    def traverse(self):
        print(f"Points: {self.points}")
        for node in self.children:
            node.traverse()


class PolarQuadTree:
    def __init__(self, polar_points):
        """
        Creates a Polar QuadTree from a point set expressed in
        polar coordinates.
        """
        assert polar_points.shape[0] >= 1
        self.root = PolarNode(np.min(polar_points[:, 0]), np.max(polar_points[:, 0]), 0.0, 2.0 * np.pi)

        # Asserts that all points can be inserted into the tree
        for point in polar_points:
            assert self.root.contains(point) == True, f"Point = ({point[0]}, {point[1]}), Node = {self.root.min_r} {self.root.max_r} {self.root.min_theta} {self.root.max_theta}"

        for point in polar_points:
            self.root.append(point)

    def render(self, ax, center=(0, 0)):
        self.root.render(ax, center)

    def traverse(self):
        self.root.traverse()

    @classmethod
    def from_cartesian_points(cls, cartesian_points):
        return cls(cartesian_to_polar(cartesian_points))


# Generate points inside Poincaré disk
points = []
num_points = 100

random.seed(42)
while len(points) != num_points:
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)

    # Test if point is inside Poincaré disk,
    # but not on the boundary.
    if x ** 2 + y ** 2 < 0.9:
        points.append((x, y))

points = np.array(points)

tree = PolarQuadTree.from_cartesian_points(points)

print(f"Number of children of root node: {len(tree.root.children)}")
print(tree.root)
for node in tree.root.children:
    print(len(node.children))
    print(node)

#tree.traverse()
print(len(tree.root.points))
print(len(set(tree.root.points)))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

# Draw Poincaré disk
poincare_boundary = plt.Circle((0, 0), 1.0, color='black')
inner_circle = plt.Circle((0, 0), tree.root.min_r, color='purple', fill=False)
outer_circle = plt.Circle((0, 0), tree.root.max_r, color='blue', fill=False)

ax.add_patch(poincare_boundary)
ax.add_patch(inner_circle)
ax.add_patch(outer_circle)
tree.render(ax)

ax.scatter(
    x=[points[:, 0]],
    y=[points[:, 1]],
    marker='o', alpha=0.9, color='white', s=3.0)

plt.show()

# import numpy as np
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
# import quads
# import random
#
#
# def cartesian_to_polar(pts):
#     """
#     Args
#         pts: ndarray containing points in Cartesian coordinates as rows.
#
#     Returns
#         ndarray: points in polar coordinates, where the first column is the
#         radius and the second column is the angle (in radians).
#     """
#     assert pts.shape[1] == 2
#     return np.column_stack((np.linalg.norm(pts, axis=1), np.arctan2(pts[:, 1], pts[:, 0])))
#
#
# def polar_equal_area_split(r_min, r_max, theta_min, theta_max, alpha=1.0):
#     """
#     Args
#         r_min, r_max (float): polar radius range of current node
#         theta_min, theta_max (float): angles (in degrees) range of the current node
#
#     Returns
#         (float, float): midpoint of radius and angle, split in half according to hyperbolic area.
#     """
#     r_mid = np.arccosh((np.cosh(alpha * r_max) + np.cosh(alpha * r_min)) / 2.0) * (1.0 / alpha)
#     theta_mid = (theta_min + theta_max) / 2.
#     return r_mid, theta_mid
#
#
# def polar_equal_length_split(r_min, r_max, theta_min, theta_max):
#     """
#     Args
#         r_min, r_max (float): polar radius range of current node
#         theta_min, theta_max (float): angles (in degrees) range of the current node
#
#     Returns
#         (float, float): midpoint of radius and angle, split in half according to Euclidean length.
#     """
#     r_mid = (r_min + r_max) / 2.
#     theta_mid = (theta_min + theta_max) / 2.
#     return r_mid, theta_mid
#
#
# class PolarNode:
#     def __init__(self, r_min, r_max, theta_min, theta_max):
#         """
#         Creates a polar node for hyperbolic box with inner radius r_min, outer radius r_max
#         and polar angle (in radians) range [min_theta, max_theta[
#         """
#         self.min_r = r_min
#         self.max_r = r_max
#         self.min_theta = theta_min
#         self.max_theta = theta_max
#
#         self.children = []
#         self.points = []  # This is inefficient, should we store only indices or better not care about it for now?
#
#     def contains(self, test_point):
#         """
#         Args
#             test_point (float, float): radius and polar angle of test point
#
#         Returns
#             True if PolarNode contains polar_point,
#             False otherwise.
#         """
#         r, theta = test_point
#         return self.min_r <= r < self.max_r and self.min_theta <= theta <= self.max_theta
#
#     def is_leaf(self):
#         return len(self.points) == 1
#
#     def is_empty(self):
#         return len(self.points) == 0
#
#
# class PolarQuadTree:
#     def __init__(self, polar_pts):
#         """
#         Creates a Polar QuadTree from a point-set expressed in
#         polar coordinates (rows are points, in polar coords).
#         """
#         assert polar_pts.shape[0] >= 1
#         # Create root, containing box spanning all angles from 0 to 2pi, but only enough radii to cover all pts
#         self.root = PolarNode(np.min(polar_pts[:, 0]), np.max(polar_pts[:, 0]), 0.0, 2.0 * np.pi)
#
#     def __insert_point(self, polar_node, point):
#         if not polar_node.contains(point):
#             return False
#
#         if polar_node.is_empty():
#             polar_node.points.append(point)
#             return True
#
#         if polar_node.is_leaf():
#             polar_node.points.append(point)  # Now it is not a leaf anymore
#
#             ## NOTE:
#             # I think that this should work if we just replace polar_equal_area_split by equal_length or
#             # something like that.
#             mid_r, mid_theta = polar_equal_area_split(polar_node.min_r, polar_node.max_r, polar_node.min_theta,
#                                                       polar_node.max_theta)
#
#             # Split node in four polar subnodes and add the subnode that contains the point
#             # to the tree
#             subnodes = [
#                 PolarNode(mid_r, max_r, min_theta, mid_theta),
#                 PolarNode(mid_r, max_r, mid_theta, max_theta),
#                 PolarNode(min_r, mid_r, mid_theta, max_theta),
#                 PolarNode(min_r, mid_r, min_theta, mid_theta)
#             ]
#
#             prev_length = len(polar_node.children)  # Sanity check, may be removed
#             points = [point, polar_node.points[0]]  # Insert new point AND previous point into subnodes
#             for subnode in subnodes:
#                 # TODO: Not tested yet, but I think this should work for the case of two points inside the same subnode (which must be recursively splitten...)
#                 for p in points:
#                     if subnode.contains(point):
#                         polar_node.children.append(subnode)
#                         self.__insert_point(subnode, point)
#
#             assert len(polar_node.children) - prev_length == 2  # Sanity check, may be removed
#
#             return True
#
#         for child in polar_node.children:
#             # Return as soon as any valid subnode is found
#             if self.__insert_point(child, point):
#                 return True
#
#         return False  # Shouldn't reach this point I guess?
#
#
# # (1) GENERATE POINTS inside Poincaré disk via rejection sampling ————————————————————————————————————————————
# # i.e. Generate point inside [-1,1]^2 square, and reject if STRICTLY outside unit disk.
# points = []
# num_points = 100
# while len(points) != num_points:
#     pt = np.random.uniform(-1, 1, 2)
#     if np.linalg.norm(pt) < 1.0:  # strict inequality
#         points.append(pt)
#
# # Covert to polar representation
# points = np.array(points)
# polar_points = cartesian_to_polar(points)
#
#
# # (2) DRAWING ——————————————————————————————————————————————————————————————————————————————————————————
# # Plotting setup (using matplotlib)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
#
# # Draw full disk boundary
# poincare_boundary = plt.Circle((0, 0), 1.0, color='black')
# ax.add_patch(poincare_boundary)
#
# # Draw hyperbolic box corresponding to root node
# inner_circle = plt.Circle((0, 0), np.min(polar_points[:, 0]), color='purple', fill=False)
# ax.add_patch(inner_circle)
# outer_circle = plt.Circle((0, 0), np.max(polar_points[:, 0]), color='blue', fill=False)
# ax.add_patch(outer_circle)
#
#
# # NOTE: better to keep all calculations in radians and *only* convert it to
# # degrees when it is time to display the result using patches.Wed
# # TODO: refactor the following into a separate class storing PolarNodes
# min_r, max_r, min_theta, max_theta = np.min(polar_points[:, 0]), np.max(polar_points[:, 0]), 0.0, 2 * np.pi
# for _ in range(8):
#     mid_r, mid_theta = polar_equal_area_split(min_r, max_r, 0.0, max_theta)
#     wedge = patches.Wedge((0, 0), r=max_r, theta1=np.degrees(mid_theta), theta2=np.degrees(max_theta),
#                           width=(max_r - mid_r), color='cyan', fill=False)
#     ax.add_patch(wedge)
#
#     max_r = mid_r
#     max_theta = mid_theta
#
# min_r, max_r, min_theta, max_theta = np.min(polar_points[:, 0]), np.max(polar_points[:, 0]), 0.0, 2 * np.pi
# for _ in range(8):
#     mid_r, mid_theta = polar_equal_area_split(min_r, max_r, 0.0, max_theta)
#     print(mid_r, mid_theta)
#     wedge = patches.Wedge((0, 0), r=mid_r, theta1=0, theta2=np.degrees(mid_theta), width=(mid_r - min_r), color='cyan',
#                           fill=False)
#     ax.add_patch(wedge)
#
#     max_r = mid_r
#     max_theta = mid_theta
#
# ax.scatter(
#     x=[points[:, 0]],
#     y=[points[:, 1]],
#     marker='o', alpha=0.9, color='white', s=1.0)
#
# plt.show()
