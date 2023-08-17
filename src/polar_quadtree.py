import numpy as np
from scipy.constants import c
import matplotlib.patches as patches

def cartesian_to_polar(points):
    """
    Args
        points: ndarray containing points in Cartesian coordinates.
    
    Returns
        ndarray: points in polar coordinates, where the first column is the
        radius and the second column is the angle (in radians).
    """
    assert points.shape[1] == 2
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles[angles < 0] += 2.0 * np.pi # Fix cases where arctan2 < 0
    return np.column_stack((np.sqrt(points[:, 0]**2 + points[:, 1]**2), angles))

def polar_to_cartesian(points):
    x = points[:, 0] * np.cos(points[:, 1])
    y = points[:, 0] * np.sin(points[:, 1])
    return np.column_stack((x, y))

def sq_norm(x, y):
    return x ** 2 + y ** 2

def poincare_to_klein(c, sq_n):
    return 2 * c / (1 + sq_n)

def klein_to_poincare(c, sq_n):
    return c / (1 + np.sqrt(1 - sq_n))

def lorentz_factor(sq_n):
    return 1 / np.sqrt(1 - sq_n)

def hyperbolic_distance(u, v, convert=False):
    if convert == True:
        u = polar_to_cartesian(np.array(u).reshape(-1, 2))
        v = polar_to_cartesian(np.array(v).reshape(-1, 2))
    norm_uv = np.linalg.norm(u - v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    assert (1 - norm_u ** 2) > 0.0
    assert (1 - norm_v ** 2) >= 0.0, f"v = {v} Norm of v = {norm_v}"
    assert (1 - norm_u ** 2) * (1 - norm_v ** 2) >= 0.0

    dist = np.arccosh(1 + 2 * (norm_uv ** 2 / ((1 - norm_u ** 2) * (1 - norm_v ** 2))))
    
    assert np.isscalar(dist)
    return dist

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
    worst_case_counter = 0

    def __init__(self, min_r, max_r, min_theta, max_theta):
        """
        Creates a polar node with inner radius min_r, outer radius max_r
        and polar angle (in radians) range [min_theta, max_theta[
        """
        self.min_r = min_r
        self.max_r = max_r
        self.min_theta = min_theta
        self.max_theta = max_theta

        self.children = []
        self.points = [] 
        self.indices = []
        self.center_of_mass = np.array([0.0, 0.0])
        self.update_center_of_mass()
        self.max_node_radius = self.max_radius()

    def max_radius(self):
        max_arc = hyperbolic_distance((self.max_r, self.min_theta), (self.max_r, self.max_theta), True)
        diagonal = hyperbolic_distance((self.min_r, self.min_theta), (self.max_r, self.max_theta), True)
        return max(max_arc, diagonal)


    def contains(self, polar_point):
        """
        Returns True if PolarNode contains polar_point,
        False otherwise. Assumes that polar_point stores
        the radius and the polar angle of the point in a tuple
        or numpy array with two columns.
        """
        r, theta = polar_point
        return r >= self.min_r and r <= self.max_r and theta >= self.min_theta and theta <= self.max_theta

    def is_leaf(self):
        return len(self.points) == 1

    def is_empty(self):
        return len(self.points) == 0
    
    def append(self, idx, point):
        if not self.contains(point):
            return False
        
        if self.is_empty():
            self.points.append((point[0], point[1]))
            self.indices.append(idx)
            self.update_center_of_mass()
            return True
        
        if self.is_leaf():
            assert len(self.children) == 0
            self.points.append((point[0], point[1])) # Not a leaf anymore
            self.indices.append(idx)
            self.update_center_of_mass()
            
            ## NOTE:
            # I think that this should work if we just replace polar_equal_area_split by equal_length or
            # something like that.
            mid_r, mid_theta = polar_equal_area_split(self.min_r, self.max_r, self.min_theta, self.max_theta)
            #mid_r, mid_theta = polar_equal_length_split(self.min_r, self.max_r, self.min_theta, self.max_theta)
            
            assert mid_r > self.min_r and mid_r < self.max_r
            assert mid_theta > self.min_theta and mid_theta < self.max_theta
            
            # Split node in four polar subnodes and add the subnode that contains the point
            # to the tree
            subnodes = [
                PolarNode(mid_r, self.max_r, self.min_theta, mid_theta), 
                PolarNode(mid_r, self.max_r, mid_theta, self.max_theta), 
                PolarNode(self.min_r, mid_r, mid_theta, self.max_theta), 
                PolarNode(self.min_r, mid_r, self.min_theta, mid_theta)
            ]
            
            points = [point, self.points[0]] # Insert new point AND previous point into subnodes
            indices = [idx, self.indices[0]]
            
            for idx, p in zip(indices, points):
                for subnode in subnodes:
                    if subnode.append(idx, p):
                        continue
            
            for subnode in subnodes:
                self.children.append(subnode)
                #if not subnode.is_empty():
                    #self.children.append(subnode)
            
            return True
        
        # If not a leaf node, recursively tries to add the point to its children.
        # I think that we can stop the search at the first children that accepts
        # the point.
        
        assert len(self.children) > 0 # If it is not a leaf, then it must have children nodes
        self.points.append((point[0], point[1]))
        self.indices.append(idx)
        self.update_center_of_mass()
        for node in self.children:
            if node.append(idx, point):
                return True
            
        return False
    
    def update_center_of_mass(self):
        if len(self.points) == 0:
            return
        cartesian_points = polar_to_cartesian(np.array(self.points).reshape(-1, 2))
        num = 0
        denom = 0
        for point in cartesian_points:
            klein_point = 2.0 * point / (1.0 + np.dot(point, point)) # v_ij
            temp_norm = sq_norm(klein_point[0], klein_point[1])
            temp_lorentz = lorentz_factor(temp_norm)
            temp_norm = sq_norm(self.center_of_mass[0], self.center_of_mass[1])
            num += temp_lorentz * klein_point
            denom += temp_lorentz
        self.center_of_mass = num/denom
        temp_norm = sq_norm(self.center_of_mass[0], self.center_of_mass[1])
        self.center_of_mass[0] = klein_to_poincare(self.center_of_mass[0], temp_norm)
        self.center_of_mass[1] = klein_to_poincare(self.center_of_mass[1], temp_norm)


    def __str__(self):
        return f"Radius range: [{self.min_r}, {self.max_r}]; Angle range: [{self.min_theta}, {self.max_theta}]"
    
    def render(self, ax, center=(0, 0)):
        wedge = patches.Wedge(center, r=self.max_r, theta1=np.degrees(self.min_theta), theta2=np.degrees(self.max_theta), width=(self.max_r - self.min_r), color='cyan', fill=False)
        ax.add_patch(wedge)
        
        for node in self.children:
            node.render(ax, center)
    
    def traverse(self):
        print(f"Points: {self.points}")
        for node in self.children:
            node.traverse()

    def query(self, anchor_point, target_polar_point, target_point, threshold):
        if not self.contains(target_polar_point):
            return [], []

        dist = hyperbolic_distance(anchor_point, self.center_of_mass)

        if self.is_leaf():
            PolarNode.worst_case_counter += 1
            return self.indices, [dist]
                
        # Approximate and prune
        if self.max_node_radius / (dist**2) < threshold:
            return self.indices, [dist]*len(self.indices)
                    
        # Split
        indices, distances = [], []
        for node in self.children:
            node_indices, node_dists = node.query(anchor_point, target_polar_point, target_point, threshold)                
            indices = indices + node_indices
            distances = distances + node_dists
            
            if len(node_indices) > 0:
                return indices, distances
        
        return indices, distances

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
        
        for idx, point in enumerate(polar_points):
            self.root.append(idx, point)
    
    @classmethod
    def from_cartesian_points(cls, cartesian_points):
        return cls(cartesian_to_polar(cartesian_points))

    def render(self, ax, center=(0, 0)):
        self.root.render(ax, center)
    
    def traverse(self):
        self.root.traverse()

    def query(self, anchor_point, target_polar_point, target_point, threshold=0.5):
        assert self.root.contains(target_polar_point)
        return self.root.query(anchor_point, target_polar_point, target_point, threshold)

