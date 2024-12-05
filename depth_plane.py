import numpy as np


def random_plane_around(p:np.ndarray):

  normal = np.random.randn(3)
  normal /= np.linalg.norm(normal)

  if normal[2] < 0:
    normal *= -1

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



fx, fy = 150, 150
cx, cy = 100, 100

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])



center = np.array([0, 0, 3])
# p = random_plane_around(center)

# points = random_points_on_plane(p, center)
# print(distance_to_plane(p, points))


# uv, d  = project_points(K, points)


# rays = (uv - np.array([cx, cy])) / np.array([fx, fy])
# rays = np.hstack([rays, np.ones((len(rays), 1))])

# depth = p[3] / -rays.dot(p[:3])
# print(d)
# print(depth)



def ray_intersects_plane(ray_origin, ray_direction, plane):
    # Calculate the dot product of the ray direction and the plane normal
    dot_product = np.dot(ray_direction, plane[:3])

    # Check if the ray is parallel to the plane
    if dot_product == 0:
        return None  # Ray is parallel to the plane, no intersection

    # Calculate the intersection point - use dot product
    t = -(np.dot(ray_origin, plane[:3]) + plane[3]) / dot_product

    if t < 0:
        return None
    
    intersection = ray_origin + t * ray_direction
    return intersection

    
planes = [random_plane_around(center) for _ in range(7)]

uv = np.array([160, 160])

ray = np.linalg.inv(K) @ np.array([*uv, 1])

print(ray)

origin = np.array([0, 0, 0])

intersecting = []
depths = []
for plane in planes:
    depth = ray_intersects_plane(origin, ray / np.linalg.norm(ray), plane)
    
    if depth is not None:
        intersecting.append(plane)
        depths.append(depth[2])


mean_plane = np.mean(intersecting, axis=0)
# normalize
mean_plane /= np.linalg.norm(mean_plane[:3])
print(mean_plane)

depth_mean = ray_intersects_plane(origin, ray / np.linalg.norm(ray), mean_plane)
print(depth_mean[2])

print(np.mean(depths))
print(planes)


# x = np.array(intersecting)[:3] @ mean_plane[:2]  
print(np.array(intersecting)[:, :3] @ mean_plane[:3]) 
