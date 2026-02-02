'''
compute rigid rotation matrix and translation vector 
between predict atom position and true atom position
'''

import torch
import numpy as np
from scipy import linalg
def rigid_transform_3D_tensor_version(A, B):

    '''
    Input:
        - A(np.matrix):points set of rigid A
        - B(np.matrix):points set of rigid B
    Output:
        - R(np.matrix):rotation matrix
        - t(np.matrix):translation vector
    '''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def rigid_transform_3D2(A, B):
    '''change data type
    Input:
        - A(tensor):points set of rigid A
        - B(tensor):points set of rigid B
    Output:
        - R(tensor):rotation matrix
        - t(tensor):translation vector
    '''
    A = A.detach()
    B = B.detach()
    A = A.permute(1, 0).numpy()
    B = B.permute(1, 0).numpy()
    R,t = rigid_transform_3D_tensor_version(A, B)
    R = torch.from_numpy(R).permute(1,0)
    t = torch.from_numpy(t).permute(1,0)
    return R, t



# transform protein, target is a single protein
def transform_pred_coor_to_label_coor_3(pred_coor, target_coor, args):
    '''
    Input:
        - pred_coor(tensor):predict atom position
        - target_coor(tensor):true atom position
        - args(object):arguments
    Output:
        - new_A_1(tensor):rotated predict atom position
    '''
    A_1 = pred_coor.to('cpu')
    B_1 = target_coor.to('cpu')

    ret_R_1, ret_t_1 = rigid_transform_3D2(A_1, B_1)

    new_A_1 = torch.matmul(A_1, ret_R_1) + ret_t_1

    return new_A_1.to(args.device)