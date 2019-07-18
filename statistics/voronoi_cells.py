from math import pi

import cv2
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
import scipy.spatial as sp


def density(features, boundary):
    voronoi_cell_area, shape_factor, on_edge = calculate(features[['x', 'y']].values, boundary)
    particle_area = (features.r.mean() ** 2 * pi)
    features['density'] = (particle_area / voronoi_cell_area).astype('float32')
    features['shape_factor'] = shape_factor.astype('float32')
    features['on_edge'] = np.array(on_edge, dtype='bool')
    return features


def calculate(particles, boundary):
    # boundary = Polygon(boundary)
    vor = sp.Voronoi(particles[:, :2])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    inside = find_points_inside(vertices, boundary)
    vertices = np.float32(vertices)
    polygons, on_edge = get_polygons(regions, vertices, inside)
    polygons = intersect_all_polygons(polygons, boundary, on_edge)
    area, shape_factor = area_and_shapefactor(polygons)
    # plot_polygons(polygons)
    return area, shape_factor, on_edge


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def find_points_inside(vertices, boundary):
    # path = mpath.Path(boundary.exterior.coords)
    path = mpath.Path(boundary)
    flags = path.contains_points(vertices)
    return flags


def get_polygons(regions, vertices, inside):
    on_edge = [not all(inside[r]) for r in regions]
    polygons = [SimplePolygon(vertices[r]) for r in regions]
    return polygons, on_edge


def intersect_all_polygons(polygons, boundary, on_edge):
    return [poly.intersection(boundary) if edge else poly
            for poly, edge in zip(polygons, on_edge)]


def area_and_shapefactor(polygons):
    area = np.array([p.area for p in polygons])
    sf = np.array([p.length for p in polygons])**2 / (4 * pi * area)
    return area, sf


def plot_polygons(polygons):
    plt.figure()
    for poly in polygons:
        # coords = np.array(poly.exterior.coords)
        coords = poly.vertices
        plt.fill(coords[:, 0], coords[:, 1])
    plt.show()


class SimplePolygon:
    def __init__(self, vertices):
        self.area = cv2.contourArea(vertices)
        self.length = cv2.arcLength(vertices, True)
        self.vertices = vertices

    def intersection(self, boundary):
        vertices = self.vertices*1000
        boundary = boundary*1000
        pc = pyclipper.Pyclipper()
        pc.AddPath(boundary, pyclipper.PT_CLIP, True)
        pc.AddPath(vertices, pyclipper.PT_SUBJECT, True)
        solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        solution = np.array(solution, dtype=np.float32)/1000
        solution = np.squeeze(solution)
        return SimplePolygon(solution)

    def area(self):
        return cv2.contourArea(self.vertices)

    def length(self):
        return cv2.arcLength(self.vertices, True)

if __name__ == "__main__":
    from Generic import filedialogs
    from ParticleTracking import dataframes, statistics
    file = filedialogs.load_filename()
    data = dataframes.DataStore(file, load=True)
    calc = statistics.PropertyCalculator(data)
    calc.density(multiprocess=True)
    print(data.df.head())
    print(data.metadata)