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
from featurePnP import optimization
from featurePnP import losses 
from datasets import robotcar_corr_dataset

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
        
        self._featurePnP = optimization.FeaturePnP(iterations=100, device=device, loss_fn=losses.squared_loss, lambda_=100, verbose=True) 

    def _compute_sparse_reference_hypercolumn(self, reference_image,
                                              local_reconstruction,
                                              return_dense=False):
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
            keypoint_association.fast_sparse_keypoint_descriptor(
                [local_reconstruction.points_2D.T], # here need to be adjusted to image resolution. 
                dense_keypoints, reference_dense_hypercolumn)[0]
        if return_dense:
            return reference_sparse_hypercolumns, cell_size, reference_dense_hypercolumn
        else:
            return reference_sparse_hypercolumns, cell_size
    
    def run_corr(self):
        self._corr = robotcar_corr_dataset.RobotcarCorrDataset(
            root="/local/home/lixxue/gnnet/robotcar_data/",
            image_folder="/local/home/lixxue/Downloads/gn_net_data/robotcar/images/",
            pair_info_folder="gt",
            queries_folder="query_all",
            robotcar_weather_all=False,
            robotcar_weather='sun',
            img_scale=1) 

        pose_errors = np.zeros((len(self._corr), 2))
        num_inliers = np.zeros((len(self._corr)))

        output = []
        tqdm_bar = tqdm(range(len(self._corr)), total=len(self._corr),
                        unit='images', leave=True)
        for i in tqdm_bar:
            reference_image = str(self._corr._data['image_pairs_name']['a'][i])
            query_image = str(self._corr._data['image_pairs_name']['b'][i])
            if reference_image not in self._filename_to_local_reconstruction:
                continue
            if query_image not in self._filename_to_intrinsics:
                continue
            query_dense_hypercolumn, _= self._network.compute_hypercolumn(
                [query_image], to_cpu=False, resize=True)
            channels, width, height = query_dense_hypercolumn.shape[1:]
            query_dense_hypercolumn_copy = query_dense_hypercolumn.clone().detach()
            query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
                (channels, -1))
            local_reconstruction = \
                self._filename_to_local_reconstruction[str(reference_image)]
            reference_sparse_hypercolumns, cell_size, reference_dense_hypercolumn = \
                self._compute_sparse_reference_hypercolumn(
                    reference_image, local_reconstruction, return_dense=True)
            # print(self._corr._data['corres_pos_all']['b'][i][:5])
            # print(self._corr._data['corres_pos_all']['b'][i].max())
            # print(len(self._corr._data['corres_pos_all']['b'][i]))
            # print(local_reconstruction.points_2D[:5])
            # print(local_reconstruction.points_2D.max())
            # print(len(local_reconstruction.points_2D))
            # Perform exhaustive search
            matches_2D, mask = exhaustive_search.exhaustive_search(
                query_dense_hypercolumn,
                reference_sparse_hypercolumns,
                Image.open(reference_image).size[::-1],
                [width, height],
                cell_size)

            # Solve PnP
            points_2D = np.reshape(
                matches_2D.cpu().numpy()[mask], (-1, 1, 2))
            points_3D = np.reshape(
                local_reconstruction.points_3D[mask], (-1, 1, 3))
            distortion_coefficients = \
                local_reconstruction.distortion_coefficients
            # Should use query's intrinsics?
            query_intrinsics = self._filename_to_intrinsics[query_image] 
            intrinsics = local_reconstruction.intrinsics
            # print(query_intrinsics)
            prediction = solve_pnp.solve_pnp(
                points_2D=points_2D,
                points_3D=points_3D,
                intrinsics=intrinsics,
                distortion_coefficients=distortion_coefficients,
                reference_filename=reference_image,
                reference_2D_points=local_reconstruction.points_2D[mask],
                reference_keypoints=None)
            predictions = [prediction]
            reference_prediction = self._nearest_neighbor_prediction(
                    reference_image)
            gt_pose = self._corr._data['poses'][i]
            pred_pose = prediction.matrix @ np.linalg.inv(reference_prediction.matrix)
            dr, dt = losses.pose_error_mat(gt_pose, pred_pose)
            pose_errors[i, 0] = dr
            pose_errors[i, 1] = dt
            num_inliers[i] = prediction.num_inliers
            print(pose_errors[i])
            # print(prediction.matrix)
            # print(reference_prediction.matrix)
            # print(prediction.matrix @ np.linalg.inv(reference_prediction.matrix))
            # print(self._corr._data['poses'][i])
            # return
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

        np.save("retrieval_error_sun", pose_errors)
        np.save("retrieval_num_inliers_sun", num_inliers)
        return output
            

    def run(self):
        """Run the sparse-to-dense pose predictor."""

        print('>> Generating pose predictions using sparse-to-dense matching...')
        output = []
        tqdm_bar = tqdm(enumerate(self._ranks.T), total=self._ranks.shape[1],
                        unit='images', leave=True)
        for i, rank in tqdm_bar:

            # Compute the query dense hypercolumn
            query_image = self._dataset.data['query_image_names'][i]
            # print(query_image.split('/')[-1])
            if query_image not in self._filename_to_intrinsics:
                continue
            query_dense_hypercolumn, _= self._network.compute_hypercolumn(
                [query_image], to_cpu=False, resize=True)
            channels, width, height = query_dense_hypercolumn.shape[1:]
            # lixin: used for featurePnP
            query_dense_hypercolumn_copy = query_dense_hypercolumn.clone().detach()
            query_dense_hypercolumn = query_dense_hypercolumn.squeeze().view(
                (channels, -1))
            predictions = []

            for j in rank[:self._top_N]:
                
                # Compute dense reference hypercolumns
                nearest_neighbor = self._dataset.data['reference_image_names'][j]
                local_reconstruction = \
                    self._filename_to_local_reconstruction[nearest_neighbor]
                reference_sparse_hypercolumns, cell_size, reference_dense_hypercolumn = \
                    self._compute_sparse_reference_hypercolumn(
                        nearest_neighbor, local_reconstruction, return_dense=True)
                # lixin: get reference image's pose:
                reference_prediction = self._nearest_neighbor_prediction(
                        nearest_neighbor)


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
                # Should use query's intrinsics?
                query_intrinsics = self._filename_to_intrinsics[query_image] 
                intrinsics = local_reconstruction.intrinsics
                # print(query_intrinsics)
                # print(intrinsics)
                # assert(np.allclose(query_intrinsics[0], intrinsics))
                # check if distortion_coefficients always zero
                # if use distortions in the query_intrinsics, should be always zero?
                # print(len(self._filename_to_intrinsics))
                # print(len(self._filename_to_local_reconstruction))
                assert(np.allclose(distortion_coefficients, 0))
                prediction = solve_pnp.solve_pnp(
                    points_2D=points_2D,
                    points_3D=points_3D,
                    intrinsics=intrinsics,
                    distortion_coefficients=distortion_coefficients,
                    reference_filename=nearest_neighbor,
                    reference_2D_points=local_reconstruction.points_2D[mask],
                    reference_keypoints=None)

                # Perform feature-metric PnP
                self._featurePnP(
                    query_prediction=prediction, 
                    reference_prediction=self._nearest_neighbor_prediction(nearest_neighbor), 
                    local_reconstruction=local_reconstruction,
                    mask=mask,
                    query_dense_hypercolumn=query_dense_hypercolumn_copy, 
                    reference_dense_hypercolumn=reference_dense_hypercolumn,
                    query_intrinsics=query_intrinsics[0],
                    size_ratio=cell_size[0], 
                    R_gt=None, 
                    t_gt=None)
                
                # print("number of inliers: ", len(mask[mask==True]) )
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