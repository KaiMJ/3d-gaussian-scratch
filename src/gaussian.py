import numpy as np
from utils.matrix import quaternion_to_rotation_vectorized
from utils.colmap_data import COLMAP_Data
from scipy.spatial import KDTree


class GaussianSplatting:
    def __init__(self, colmap_data: COLMAP_Data, images_path='data/images'):
        """
        M ← SfM Points ⊲ Positions
        """
        self.points = colmap_data.points
        self.colors = colmap_data.colors

        # self.images, self.height, self.width = self.get_images(images_path)
        self.sigmas = None
        self.alphas = None

    # def get_images(self, images_path):
    #     images_data = sorted(glob(images_path + '/*.png'))
    #     with ThreadPoolExecutor() as executor: # faster loading
    #         images = list(executor.map(cv2.imread, images_data))
    #     height, width, _ = images[0].shape
    #     return images, height, width
    
    def init_attributes(self):
        """
        𝑆,𝐶, 𝐴 ← InitAttributes() ⊲ Covariances, Colors, Opacities

        https://medium.com/@yianyao1994/gaussian-splatting-part-2-representation-and-initialization-c0a036adf16e
        
        Center: Set to the point cloud locations from SfM.
        Scaling factor: Initialized as isotropic, using the mean distance to the 3-nearest neighbors.
        Rotation: No initial rotation is applied.
        Opacity: Set to 0.1.
        Spherical Harmonics (SH): Inherited from the color information in the point cloud.

        """
        # 3D covariance matrix Σ
        pass

    def init_gaussian_covariance(self):
        """
        Explanation
        Sec 4: 
            Directly optimizing covariance matrix Σ can lead to non positive non semi-definite covariance matrices.
            Instead, we factorize Σ = 𝑅𝑆𝑆^𝑇𝑅^𝑇 [7dof], allowing anisotropic covariance and valid covariance matrices.
            For independent optimization, we store S: 3D vector for scaling and quaternion q (normalize to get valid unit quaternion)

            Convert these splats to pixel space
                W viewing transformation or extrinsics
                J Jacobian projective transformation or intrinsics
                Σ' = JW Σ W^TJ^T

        Initialization
        Sec 5.1
            "We estimate the initial covariance matrix as an isotropic Gaussian 
            with axes equal to the mean of the distance to the closest three points."

        Gradient Computation
        Appendix A
            dΣ' / ds = dΣ'/dΣ * dΣ / ds
            dΣ' / dq = dΣ'/dΣ * dΣ / dq

        Simplify Σ' = JW Σ W^TJ^T using U=JW and Σ' being (symmetric) upper left 2x2
        Σ' = U Σ U^T
        """
        points = self.points

        # non gradient implementation for now
        # Have to revist complexity of KDTree
        print(points.shape)
        kdtree = KDTree(points)
        distances, _ = kdtree.query(points, k=4)  # k=4 includes the point itself
        
        nearest_distances = distances[:, 1:4]
        mean_distances = nearest_distances.mean(axis=1)

        unit_q = np.array([1, 0, 0, 0])
        # ensure q is normalized
        q = np.repeat(unit_q[np.newaxis, :], len(points), axis=0) # (N, 4)
        r = quaternion_to_rotation_vectorized(q) # (N, 3, 3)

        # when initalizing, we sqrt since we will do (S @ ST) first
        s = mean_distances[:, None, None] * np.eye(3) # (N, 3, 3)

        # we do (S @ ST) first because it's a symmetric matrix
        # then we apply Rotation
        sigma = np.einsum('nij,nkj,nml,nlk->nim', s, s, r, r)
        return sigma