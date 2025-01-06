import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_point_cloud(points, colors, title="3D Point Cloud"):
    # Matplotlib visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    # Open3D visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=title)

def create_ellipsoids_as_one_mesh(points, sigmas, colors, sphere_resolution=5):
    """
    Usage:
    ```
        big_mesh = create_ellipsoids_as_one_mesh(points, sigmas, colors, sphere_resolution=10)
        o3d.visualization.draw_geometries([big_mesh])
    ```
    """
    base_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=sphere_resolution)
    base_sphere.compute_vertex_normals()
    
    all_vertices = []
    all_triangles = []
    all_colors = []

    base_vertices = np.asarray(base_sphere.vertices)
    base_triangles = np.asarray(base_sphere.triangles)

    vertex_count = 0
    
    for center, sigma, color in zip(points, sigmas, colors):
        scales = np.diagonal(sigma)
        transformed_vertices = base_vertices * scales + center

        all_vertices.append(transformed_vertices)
        all_colors.append(np.tile(color, (len(base_vertices), 1)))
        
        shifted_triangles = base_triangles + vertex_count
        all_triangles.append(shifted_triangles)

        vertex_count += len(base_vertices)
    
    all_vertices = np.vstack(all_vertices)
    all_triangles = np.vstack(all_triangles)
    all_colors = np.vstack(all_colors)

    big_mesh = o3d.geometry.TriangleMesh()
    big_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    big_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    big_mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
    big_mesh.compute_vertex_normals()

    return big_mesh