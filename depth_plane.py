import numpy as np


def random_plane_around(p:np.ndarray):

  normal = np.random.randn(3)
  normal /= np.linalg.norm(normal)

  d = p.dot(normal)
  return np.array([*normal, -d])

def distance_to_plane(plane, points):
  homog_points = np.hstack([points, np.ones((len(points), 1))])
  return homog_points.dot(plane)

def random_points_on_plane(plane:np.ndarray, near:np.ndarray, n_points=10, scale=1.0):
  points = np.random.randn(n_points, len(plane) - 1) * scale + near
  dist = distance_to_plane(plane, points)


  return points - (dist[:, np.newaxis] * plane[np.newaxis, :-1])

def project_points(K, points):
  p = K @ points.T 
  return (p[:2] / p[2]).T, p[2]

np.set_printoptions(precision=3, suppress=True)

center = np.array([0, 0, 3])
p = random_plane_around(center)

points = random_points_on_plane(p, center)

print(distance_to_plane(p, points))


fx, fy = 150, 150
cx, cy = 100, 100

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

print(K)  

uv, d  = project_points(K, points)
print(uv, d)

	# const float2 ray = { (pixf.x - cx) / focal_x, (pixf.y - cy) / focal_y };

rays = (uv - np.array([cx, cy])) / np.array([fx, fy])
rays = np.hstack([rays, np.ones((len(rays), 1))])

print(rays.dot(p[:3]))

