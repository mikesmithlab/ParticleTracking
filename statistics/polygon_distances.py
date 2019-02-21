from shapely.geometry import Polygon, Point


def to_points(poly, points):
    poly = Polygon(poly)
    distance = [poly.exterior.distance(Point(p)) for p in points]
    return distance