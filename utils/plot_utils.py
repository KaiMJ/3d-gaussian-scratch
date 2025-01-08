import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection


def plot_points(points_2d, colors, width=960, height=540):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, s=1)
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    plt.savefig('docs/images/projected_frame_00001.png',
                bbox_inches='tight', pad_inches=0)
    plt.show()


def create_ellipsoids_as_one_mesh(points, sigmas, colors, sphere_resolution=5, std=1):
    """
    Usage:
    ```
        big_mesh = create_ellipsoids_as_one_mesh(points, sigmas, colors, sphere_resolution=10)
        o3d.visualization.draw_geometries([big_mesh])
    ```
    """
    base_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0, resolution=sphere_resolution)
    base_sphere.compute_vertex_normals()

    all_vertices = []
    all_triangles = []
    all_colors = []

    base_vertices = np.asarray(base_sphere.vertices)
    base_triangles = np.asarray(base_sphere.triangles)

    vertex_count = 0

    for center, sigma, color in zip(points, sigmas, colors):
        # for isotropic gaussian, assume eigenvalue is diagonal of sigma
        scales = np.sqrt(np.diagonal(sigma)) * std
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


def plot_covariance(points_2d, sigmas, colors, scale=1, width=960, height=540):
    # for anisotropic covariance, modify this
    eigenvals, eigenvecs = np.linalg.eigh(sigmas * scale)

    angles = np.degrees(np.arctan2(eigenvecs[:, 1, 0], eigenvecs[:, 0, 0]))
    widths = 2 * np.sqrt(eigenvals[:, 0])
    heights = 2 * np.sqrt(eigenvals[:, 1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ec = EllipseCollection(
        widths, heights, angles,
        units='x',
        offsets=points_2d,
        transOffset=ax.transData,
        facecolors=colors,
        alpha=0.1
    )

    ax.add_collection(ec)

    ax.scatter(points_2d[:, 0], points_2d[:, 1],
               c=colors, s=1, alpha=0.5)

    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    save_path = f'docs/images/projected_gaussian_00001_scaled_{scale}.png'
    plt.savefig(save_path,
                bbox_inches='tight',
                pad_inches=0)
    plt.show()
