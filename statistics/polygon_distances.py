from shapely.geometry import Polygon, Point
from tqdm import tqdm

def to_points(poly, points):
    poly = Polygon(poly)
    distance = [poly.exterior.distance(Point(p)) for p in tqdm(points, 'Edge distance')]
    return distance