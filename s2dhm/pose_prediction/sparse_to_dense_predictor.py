"""Sparse-To-Dense Predictor Class.
"""
import gin
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pose_prediction import predictor
from pose_prediction import solve_pnp
from pose_prediction import keypoint_association
from pose_prediction import exhaustive_search
from visualization import plot_correspondences

from pathlib import Path
import os

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

@gin.configurable
class SparseToDensePredictor(predictor.PosePredictor):
    """Sparse-to-dense Predictor Class.
    """
    def __init__(self, top_N: int, output_path: str, **kwargs):
        """Initialize class attributes.
        Args:
            top_N: Number of nearest neighbors to consider in the
                sparse-to-dense matching.
        """
        super().__init__(**kwargs)
        self._top_N = top_N
        self._output_path = output_path
        self._filename_to_pose = \
            self._dataset.data['reconstruction_data'].filename_to_pose
        self._filename_to_intrinsics = \
            self._dataset.data['filename_to_intrinsics']
        self._filename_to_local_reconstruction = \
            self._dataset.data['filename_to_local_reconstruction']

    def _compute_sparse_reference_hypercolumn(self, reference_image,
                                              local_reconstruction):
        """Compute hypercolumns at every visible 3D point reprojection."""
        reference_dense_hypercolumn, image_size = \
            self._network.compute_hypercolumn(
                [reference_image], to_cpu=False, resize=True)
        dense_keypoints, cell_size = keypoint_association.generate_dense_keypoints(
            (reference_dense_hypercolumn.shape[2:]),
            Image.open(reference_image).size[::-1], to_numpy=True)
        dense_keypoints = torch.from_numpy(dense_keypoints).to(device)
        # the following code is memory demanding.
        # reference_sparse_hypercolumns = \
        #     keypoint_association.fast_sparse_keypoint_descriptor(
        #         [local_reconstruction.points_2D.T], 
        #         dense_keypoints, reference_dense_hypercolumn)[0]
        reference_sparse_hypercolumns = \
            keypoint_association.my_fast_sparse_keypoint_descriptor(
                [local_reconstruction.points_2D], 
                dense_keypoints, reference_dense_hypercolumn, cell_size[0], cell_size[1])[0]

        return reference_sparse_hypercolumns, cell_size

    def run(self):
        """Run the sparse-to-dense pose predictor."""


        # save query images' hypercolumn
        # query_features = {}
        # print(">> Computing query images' features")
        # for i in tqdm(range(len(self._dataset.data['query_image_names']))):
        #     query_image = self._dataset.data['query_image_names'][i]
        #     if query_image not in self._filename_to_intrinsics:
        #         continue
        #     query_dense_hypercolumn, _ = self._network.compute_hypercolumn(
        #         [query_image], to_cpu=True, resize=True)
        #     query_dense_hypercolumn = query_dense_hypercolumn.squeeze()
        #     # query_features[query_image] = query_dense_hypercolumn
        #     query_slice = query_image.split("/")[-3]
        #     img_name = query_image.split("/")[-1]
        #     parent = Path(self._output_path, "query")
        #     if not os.path.exists(parent):
        #         os.makedirs(parent)
        #     parent = Path(parent, query_slice)
        #     if not os.path.exists(parent):
        #         os.makedirs(parent)
        #     output_filename = Path(parent, img_name)
        #     torch.save(query_dense_hypercolumn, output_filename.with_suffix('.pt'))

        # # query_slice = query_image.split("/")[-3]
        # # torch.save(query_features, Path(self._output_path, "query_features_{}.pt".format(query_slice)))

        # print(">> Computing reference images' features")
        # #reference_features = {}
        # for i in tqdm(range(len(self._dataset.data['reference_image_names']))):
        #     reference_image = self._dataset.data['reference_image_names'][i]
        #     reference_dense_hypercolumn, _ = self._network.compute_hypercolumn(
        #         [reference_image], to_cpu=True, resize=True)
        #     reference_dense_hypercolumn = reference_dense_hypercolumn.squeeze()
        #     reference_slice = reference_image.split("/")[-3]
        #     img_name = reference_image.split("/")[-1]
        #     parent = Path(self._output_path, "reference")
        #     if not os.path.exists(parent):
        #         os.makedirs(parent)
        #     parent = Path(parent, reference_slice)
        #     if not os.path.exists(parent):
        #         os.makedirs(parent)
        #     output_filename = Path(parent, img_name)
        #     torch.save(reference_dense_hypercolumn, output_filename.with_suffix('.pt'))

            # reference_feature[reference_image] = reference_dense_hypercolumn
        # torch.save(reference_features, Path(self._output_path, "reference_features_{}.pt".format(query_slice)))


        print('>> Generating pose predictions using sparse-to-dense matching...')
        output = []
        tqdm_bar = tqdm(enumerate(self._ranks.T), total=self._ranks.shape[1],
                        unit='images', leave=True)
        for i, rank in tqdm_bar:

            # Compute the query dense hypercolumn
            query_image = self._dataset.data['query_image_names'][i]
            print(query_image.split('/')[-1])
            if query_image not in self._filename_to_intrinsics:
                continue
            query_dense_hypercolumn, _= self._network.compute_hypercolumn(
                [query_image], to_cpu=False, resize=True)
            channels, width, height = query_dense_hypercolumn.shape[1:]
            query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
                (channels, -1))
            predictions = []

            ### save query_hypercolumn
            # img_name = query_image.split("/")[-1]
            # parent = Path(self._output_path, query_slice)
            # if not os.path.exists(parent):
            #     os.makedirs(parent)
            #　output_filename = Path(parent, img_name)
            #　torch.save(query_dense_hypercolumn, output_filename.with_suffix('.pt'))

            for j in rank[:self._top_N]:
            # Compute dense reference hypercolumns
                nearest_neighbor = self._dataset.data['reference_image_names'][j]
                local_reconstruction = \
                    self._filename_to_local_reconstruction[nearest_neighbor]
                reference_sparse_hypercolumns, cell_size= \
                    self._compute_sparse_reference_hypercolumn(
                        nearest_neighbor, local_reconstruction)


                # Perform exhaustive search
                matches_2D, mask = exhaustive_search.exhaustive_search(
                    query_dense_hypercolumn,
                    reference_sparse_hypercolumns,
                    Image.open(nearest_neighbor).size[::-1],
                    [width, height],
                    cell_size)

                # Solve PnP
                points_2D = np.reshape(
                    matches_2D.cpu().numpy()[mask], (-1, 1, 2))
                points_3D = np.reshape(
                    local_reconstruction.points_3D[mask], (-1, 1, 3))
                distortion_coefficients = \
                    local_reconstruction.distortion_coefficients
                intrinsics = local_reconstruction.intrinsics
                prediction = solve_pnp.solve_pnp(
                    points_2D=points_2D,
                    points_3D=points_3D,
                    intrinsics=intrinsics,
                    distortion_coefficients=distortion_coefficients,
                    reference_filename=nearest_neighbor,
                    reference_2D_points=local_reconstruction.points_2D[mask],
                    reference_keypoints=None)
                
                print("number of inliers: ", len(mask[mask==True]) )
                # If PnP failed, fall back to nearest-neighbor prediction
                if not prediction.success:
                    prediction = self._nearest_neighbor_prediction(
                        nearest_neighbor)
                    if prediction:
                        predictions.append(prediction)
                else:
                    predictions.append(prediction)
                

            if len(predictions):
                export, best_prediction = self._choose_best_prediction(
                    predictions, query_image)
                if self._log_images:
                    if np.ndim(np.squeeze(best_prediction.query_inliers)):
                        self._plot_inliers(
                            left_image_path=query_image,
                            right_image_path=best_prediction.reference_filename,
                            left_keypoints=keypoint_association.kpt_to_cv2(
                                best_prediction.query_inliers),
                            right_keypoints=keypoint_association.kpt_to_cv2(
                                best_prediction.reference_inliers),
                            matches=[(i, i) for i in range(best_prediction.num_inliers)],
                            title='Sparse-to-Dense Correspondences',
                            export_filename=self._dataset.output_converter(query_image))

                    plot_correspondences.plot_image_retrieval(
                        left_image_path=query_image,
                        right_image_path=best_prediction.reference_filename,
                        title='Best match',
                        export_filename=self._dataset.output_converter(query_image))
                    
                    # visualize features
                    best_ref_dense_hypercolumn, _= self._network.compute_hypercolumn(
                        [best_prediction.reference_filename], to_cpu=False, resize=True)
                    query_dense_hypercolumn, _= self._network.compute_hypercolumn(
                        [query_image], to_cpu=False, resize=True)
                    best_ref_dense_hypercolumn = best_ref_dense_hypercolumn.squeeze().view(
                        (channels, -1))
                    query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
                        (channels, -1))

                    # ref = best_ref_dense_hypercolumn[0,48:,:,:]  
                    # ref = ref.view((16, -1))
                    # qry = query_dense_hypercolumn[0,48:,:,:] 
                    # qry = qry.view((16, -1))
                    
                    
                    plot_correspondences.plot_feature_pca(
                        left_image_path=query_image,
                        right_image_path=best_prediction.reference_filename,
                        title='Best match feature visualization',
                        export_filename=self._dataset.output_converter(query_image),
                        left_features = query_dense_hypercolumn,
                        right_features = best_ref_dense_hypercolumn,
                        H = width,
                        W = height) # h and w are flipped in s2d code


                output.append(export)
                tqdm_bar.set_description(
                    "[{} inliers]".format(best_prediction.num_inliers))
                tqdm_bar.refresh()

        return output

    @property
    def dataset(self):
        return self._dataset