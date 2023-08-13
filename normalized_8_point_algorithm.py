import torch
import numpy as np

def normalized_eight_point_essential_matrix(img1_points, img2_points, camera_1_matrix, camera_2_matrix, device='cpu'):
    """
    img1_points: points on image 1, in shape of N * 2
    img2_points: points on image 2, in shape of N * 2
    camera_1_matrix, camera_2_matrix: camera matrix in form 3 * 3
    """
    if img1_points.shape[1] != 2 or img2_points.shape[1] != 2:
        raise ValueError('Dimention of each point in the image should be 2.')
    if img1_points.shape[0] != img2_points.shape[0]:
        raise ValueError('Number of points in image one and two should be equal.')
    if len(img1_points) < 8:
        raise ValueError('Number of corresponding points should be greater or equal to 8.')
    if camera_1_matrix.shape[0] != 3 and camera_1_matrix.shape[1] != 3:
        raise ValueError('Inputed camera matrix is not correct.')

    num_corresponding = img1_points.shape[0]

    with torch.no_grad():
        # whiten

        # convert points to homogeneous
        img1_points_hmg = np.concatenate((img1_points, np.ones(shape=(num_corresponding, 1))), axis=1) # N * 3
        img2_points_hmg = np.concatenate((img2_points, np.ones(shape=(num_corresponding, 1))), axis=1) # N * 3
        
        # convert to tensor
        img1_points_hmg = torch.tensor(data=img1_points_hmg, dtype=torch.float32, device=device) # N * 3
        img2_points_hmg = torch.tensor(data=img2_points_hmg, dtype=torch.float32, device=device) # N * 3
        camera_1_matrix_tensor = torch.tensor(data=camera_1_matrix, dtype=torch.float32, device=device) # 3 * 3
        camera_2_matrix_tensor = torch.tensor(data=camera_2_matrix, dtype=torch.float32, device=device) # 3 * 3

        # find local ray direction that passes in image 1
        # local ray direction = inverse of camera matrix * point on image in homogeneous 
        img1_lrd =  torch.matmul(camera_1_matrix_tensor.inverse(), img1_points_hmg.t()).t() # N * 3 
        # Calculate the norms of each row
        row_norms = torch.norm(img1_lrd, dim=1, keepdim=True)
        # Normalize each row by dividing by its norm
        img1_lrd_normalized = img1_lrd / row_norms
        img1_lrd = img1_lrd_normalized

        # find local ray direction that passes in image 2
        img2_lrd =  torch.matmul(camera_2_matrix_tensor.inverse(), img2_points_hmg.t()).t() # N * 3
        # Calculate the norms of each row
        row_norms = torch.norm(img2_lrd, dim=1, keepdim=True)
        # Normalize each row by dividing by its norm
        img2_lrd_normalized = img2_lrd / row_norms
        img2_lrd = img2_lrd_normalized

        # convert each correspoding local ray direction pair from
        # [x1, y1, 1] and [x2, y2, 1] to
        # [x1x2, y1x2, x2, x1y2, y1y2, y2, x1, y1, 1] in an efficient way by
        # calculating the Kronecker product for the batch
        kron_product = torch.bmm(img2_lrd.view(num_corresponding, 3, 1), img1_lrd.view(num_corresponding, 1, 3)).view(num_corresponding, -1) # N * 9
        Y = kron_product.t() # 9 * N

        # flatten essential matirx(e) can be obtained by finding left singular vector
        # corresponding to lowest singular value of SVD decomposition of Y
        U, S, Vh = torch.linalg.svd(Y, full_matrices=True)
        
        # essential matrix
        e = U[:, -1]
        E = torch.reshape(e, shape=(3, 3))

        # approximate E by a rank 2 matrix
        U, S, Vh = torch.linalg.svd(E, full_matrices=True)
        S[2] = 0.0
        E_rank2 = torch.matmul(U, torch.matmul(torch.diag(S), Vh))

        # epipole in image 1
        ep_1 = Vh[-1, :]
        if ep_1[2] != 0:
            ep_1_normalized = ep_1 / ep_1[2]
        # epipole in image 2
        ep_2 = U[:, -1]
        if ep_2[2] != 0:
            ep_2_normalized = ep_2 / ep_2[2]

        return {"essential_matrix": E_rank2.numpy(), "epipole_img_1": ep_1_normalized.numpy(), "epipole_img_2": ep_2_normalized.numpy()}


    
    



if __name__ == "__main__":
    pass