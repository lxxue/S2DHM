import torch
from torch import nn

from featurePnP.utils import (to_homogeneous, from_homogeneous, batched_eye_like,
                    skew_symmetric, so3exp_map, sobel_filter)
from featurePnP.losses import scaled_loss, squared_loss, pose_error

from PIL import Image
import scipy.io
import numpy as np
import logging

def optimizer_step(g, H, lambda_=0):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
    """
    if lambda_:  # LM instead of GN
        D = (H.diagonal(dim1=-2, dim2=-1) + 1e-9).diag_embed()
        H = H + D*lambda_
    try:
        P = torch.inverse(H)
    except RuntimeError as e:
        logging.warning(f'Determinant: {torch.det(H)}')
        raise e
    delta = -(P @ g[..., None])[..., 0]
    return delta

class FeaturePnP(nn.Module):
    def __init__(self, iterations, device, loss_fn=squared_loss, lambda_=0.01, verbose=False):
        super().__init__()
        self.iterations = iterations
        self.device = device
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.lambda_ = lambda_


    def forward(self, 
                query_prediction, 
                reference_prediction, 
                local_reconstruction, 
                mask,
                query_dense_hypercolumn, 
                reference_dense_hypercolumn,
                query_intrinsics, 
                size_ratio,
                R_gt=None, 
                t_gt=None):
        # TODO: take distCoeffs into account using cv2.undistortPoints
        # TODO: understand scale here
        with torch.no_grad():
            q_matrix = torch.FloatTensor(query_prediction.matrix).to(self.device)
            q_matrix_inv = q_matrix.inverse()
            r_matrix = torch.FloatTensor(reference_prediction.matrix).to(self.device)
            r_matrix_inv = r_matrix.inverse()

            # TODO: check to use inverse or not?
            # pts0 = r_matrix * 3D
            # pts1 = q_matrix * 3D
            # from pts0 to pts1: q_matrix @ r_matrix_inv
            relative_matrix = q_matrix @ r_matrix_inv
            # relative_matrix = q_matrix_inv @ r_matrix

            R_init = relative_matrix[:3, :3]
            t_init = relative_matrix[3, :3]
            imgf0 = reference_dense_hypercolumn
            imgf1 = query_dense_hypercolumn
            imgf1_gx, imgf1_gy = sobel_filter(imgf1) 
            imgf1_gx = imgf1_gx.squeeze()
            imgf1_gy = imgf1_gy.squeeze()
            imgf0 = imgf0.squeeze()
            imgf1 = imgf1.squeeze()
            K0 = torch.FloatTensor(local_reconstruction.intrinsics).to(self.device)
            K1 = torch.FloatTensor(query_intrinsics).to(self.device)

            lambda_ = self.lambda_

            # 0 is reference image, 1 is query image
            # already in screen coordinates
            pts_2d_0 = torch.FloatTensor(local_reconstruction.points_2D[mask]).to(self.device) # N x 2  
            img0_idx = torch.floor(pts_2d_0 / size_ratio).type(torch.LongTensor)
            extracted_feat0 = (imgf0[:, img0_idx[:, 1], img0_idx[:, 0]]).transpose(0, 1)

            # pts_3d_0 = to_homogeneous(local_reconstruction.points_3D[mask]) @ r_matrix_inv.T
            pts_3d_0 = from_homogeneous(
                to_homogeneous(
                torch.FloatTensor(local_reconstruction.points_3D[mask])).to(self.device) 
                @ r_matrix.T)

            R = R_init
            t = t_init

            scale = torch.ones((pts_2d_0.shape[0],)).type(torch.FloatTensor).to(self.device)
            # TODO: add early stop for LM
            for i in range(self.iterations):
                pts_3d_1 = pts_3d_0 @ R + t
                pts_2d_1 = from_homogeneous(pts_3d_1 @ K1.T)
                img1_idx = torch.floor(pts_2d_1 / size_ratio).type(torch.LongTensor).to(self.device)
                extracted_feat1 = (imgf1[:, img1_idx[:, 1], img1_idx[:, 0]]).transpose(0, 1)

                error = extracted_feat1 - extracted_feat0
                cost = (error**2).sum(-1)
                # TODO: understand what this scaled_loss is
                cost, weights, _ = scaled_loss(cost, self.loss_fn, scale)
                if i == 0:
                    prev_cost = cost.mean(-1)
                if self.verbose:
                    print('Iter ', i, cost.mean().item())

                # calculate gradient for LM
                J_p_T = torch.cat([
                    batched_eye_like(pts_3d_1, 3), -skew_symmetric(pts_3d_1,)
                ], -1)

                shape = pts_3d_1.shape[:-1]
                o, z = pts_3d_1.new_ones(shape), pts_3d_1.new_zeros(shape)

                fx = K1[0, 0]
                fy = K1[1, 1]

                J_h_p = torch.stack([
                    o*fx,   z,      -fx*pts_3d_1[..., 0] / pts_3d_1[..., 2],
                    z,      o*fy,   -fy*pts_3d_1[..., 1] / pts_3d_1[..., 2],
                ], dim=-1).reshape(shape+(2, 3)) / pts_3d_1[..., 2, None, None] / size_ratio

                J_e_hx = (imgf1_gx[:, img1_idx[:, 1], img1_idx[:, 0]]).transpose(0, 1)
                J_e_hy = (imgf1_gy[:, img1_idx[:, 1], img1_idx[:, 0]]).transpose(0, 1)
                J_e_h = torch.stack([J_e_hx, J_e_hy], dim=-1)
                J_e_T = J_e_h @ J_h_p @ J_p_T

                Grad = torch.einsum('...ijk,...ij->...ik', J_e_T, error)
                Grad = weights[..., None] * Grad
                Grad = Grad.sum(-2)  # Grad was ... x N x 6
                J = J_e_T
                Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)
                Hess = weights[..., None, None] * Hess
                # Hess = weights[..., None]  * Hess
                Hess = Hess.sum(-3)  # Hess was ... x N x 6 x 6

                delta = optimizer_step(Grad, Hess, lambda_)
                if torch.isnan(delta).any():
                    logging.warning('NaN detected, exit')
                    break
                dt, dw = delta[..., :3], delta[..., 3:6]
                dr = so3exp_map(dw)
                R_new = dr @ R
                t_new = dr @ t + dt

                new_pts_3d_1 = pts_3d_0 @ R_new.T + t_new
                new_pts_2d_1 = from_homogeneous(new_pts_3d_1 @ K1.T)
                new_img1_idx = torch.floor(new_pts_2d_1 / size_ratio).type(torch.LongTensor).to(self.device)
                new_extracted_feat1 = (imgf1[:, new_img1_idx[:, 1], new_img1_idx[:, 0]]).transpose(0, 1)
            

                new_error = new_extracted_feat1 - extracted_feat0 
                new_cost = (new_error**2).sum(-1)
                new_cost = scaled_loss(new_cost, self.loss_fn, scale)[0].mean()

                lambda_ = np.clip(lambda_ * (10 if new_cost > prev_cost else 1/10),
                                  1e-6, 1e2)
                if new_cost > prev_cost:  # cost increased
                    continue
                prev_cost = new_cost
                R, t = R_new, t_new

                if R_gt is not None and t_gt is not None:
                    if self.verbose:
                        gt_error = pose_error(R, t, R_gt, t_gt)
                        print('Pose error:', *gt_error)
            return R, t

   





        

