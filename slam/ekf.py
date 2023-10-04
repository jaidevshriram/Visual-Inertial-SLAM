import torch
import numpy as np

import matplotlib.pyplot as plt

from utils.pose import *

def is_psd(x):

    eigvals = torch.linalg.eigvals(x)

    out = True
    for i, eigval in enumerate(eigvals):
        if eigval.real <= 0:
            print("Not PSD", i, eigval)
            out = False
            break

    return out

class EKFSLAM:

    def __init__(self, n_landmarks=10, imu_T_cam=torch.eye(4), device='cpu'):
        
        self.device = device
        self.landmark_cov_noise = 2
        self.robot_cov_noise = 1e-5

        self.pos_noise = 1e-5
        self.ang_noise = 1e-5

        self.measurement_noise = 5

        # Initialize robot pose
        self.mu = torch.eye(4).double().to(self.device)

        # Initialize landmark and robot covariance
        self.cov = torch.eye(3 * n_landmarks + 6).double().to(self.device) # (3*n_landmarks + 6, 3*n_landmarks + 6)
        self.cov[-6:, -6:] *= self.robot_cov_noise # Robot Pose Noise
        self.cov[:-6, :-6] *= self.landmark_cov_noise # Landmark Noise

        # self.cov = torch.eye(3 * n_landmarks).double().to(self.device) # (3*n_landmarks + 6, 3*n_landmarks + 6)
        # self.cov *= self.landmark_cov_noise # Landmark Noise

        # Initialise the landmarks
        self.landmarks = torch.zeros((3, n_landmarks)).double().to(self.device)

        self.landmark_seen = torch.zeros(n_landmarks, dtype=torch.bool).to(self.device)
        self.n_landmarks = n_landmarks

        self.cam_to_imu = imu_T_cam

    def predict(self, u, dt):
        
        # Update robot pose
        # self.mu = self.exp(-dt * axangle2twist(u)) @ self.mu
        self.mu = self.mu @ self.exp(dt * axangle2twist(u))
        # self.cov = self.exp(-dt * axangle2adtwist(u)) @ self.cov @ self.exp(-dt * axangle2adtwist(u)).transpose(0, 2, 1) + W - old, legacy code

        F = self.exp(-dt * axangle2adtwist(u))[0]

        W = torch.eye(6).double().to(self.device)
        W[:3, :3] *= self.pos_noise # Pos Noise
        W[3:, 3:] *= self.ang_noise # Rot Noise

        # Update robot covariance
        robot_cov = self.cov[-6:, -6:]
        robot_cov = F @ robot_cov @ F.T + W
        self.cov[-6:, -6:] = robot_cov

        # Update LR covariance
        lr_cov = self.cov[:-6, -6:]
        lr_cov = lr_cov @ F.T
        self.cov[:-6, -6:] = lr_cov

        # Update RL covariance
        rl_cov = self.cov[-6:, :-6]
        rl_cov = F @ rl_cov
        self.cov[-6:, :-6] = rl_cov

        # Assert that covariance is symmetric
        # assert torch.sum(torch.abs(self.cov - self.cov.T)) < 1e-3, "Covariance is not symmetric" + str(torch.sum(torch.abs(self.cov - self.cov.T)))        

    def init_and_obtain_landmarks(self, features, K, baseline):

        # Filter out features that are not visible
        features_idxs = (features != -1).all(axis=0)
        not_seen = (~self.landmark_seen).to(self.device)
        features_idxs_unseen = features_idxs & not_seen

        # Create landmarks from these observations
        features = features[:, features_idxs_unseen]

        # Use the stereo camera to triangulate the landmarks
        new_landmark_points = torch.ones((4, features.shape[1])).double().to(self.device)

        ul, vl = features[0, :], features[1, :]
        ur, vr = features[2, :], features[3, :]

        depth = baseline * K[0, 0] / (ul - ur)
        new_landmark_points[0, :] = (ul - K[0, 2]) * depth / K[0, 0]
        new_landmark_points[1, :] = (vl - K[1, 2]) * depth / K[1, 1]
        new_landmark_points[2, :] = depth

        # if depth.shape[0] > 0:
            # print("Depth range", torch.min(depth), torch.max(depth))

        # Only initialise points that are < 100m away
        features_idxs_unseen[features_idxs_unseen.clone()] = (depth < 200) & (depth > 0)
        new_landmark_points = new_landmark_points[:, (depth < 200) & (depth > 0)]

        # Convert to camera frame
        new_landmark_points = self.get_cam_to_world() @ new_landmark_points

        self.landmarks[:, features_idxs_unseen] = new_landmark_points[:3, :]

        # Update landmark seen
        self.landmark_seen[features_idxs_unseen] = True       

        # Only return landmarks that are visible and were seen before for update
        features_idxs = features_idxs & ~not_seen

        # Only select landmarks < 100m away

        if features_idxs.sum() != 0:
            landmarks_hom = torch.vstack((self.landmarks[:, features_idxs], torch.ones(self.landmarks[:, features_idxs].shape[1]).double().to(self.device)))
            depth = self.get_world_to_cam() @ landmarks_hom

            # plt.hist(depth[2, :].cpu().numpy(), bins=100)
            # plt.show()

            # features_idxs[features_idxs.clone()] = depth[2, :] < 75

        return self.landmarks[:, features_idxs], torch.argwhere(features_idxs).flatten(), features_idxs

    def update(self, features, K, baseline, imu_T_cam):

        # Initialise landmarks if not already initialised
        landmarks_from_pose, selected_landmarks, _ = self.init_and_obtain_landmarks(features, K, baseline)
        num_landmarks_seen = landmarks_from_pose.shape[1]

        if num_landmarks_seen == 0:
            print("No landmarks seen")
            return
        # else:
        # return

        # Convert to homogenous coordinates
        landmarks_from_pose_hom = torch.vstack((landmarks_from_pose, torch.ones(landmarks_from_pose.shape[1]).double().to(self.device)))

        # Stereo camera intrinsics
        M = torch.zeros((4, 4)).double().to(self.device)
        M[:2, :3] = K[:2, :] # First two rows of K for ul, vl
        M[2:, :3] = K[:2, :] # First two rows of K for ur, vr
        M[2, 3] = -K[0, 0] * baseline

        # Predicted landmark positions in the two cameras
        predicted_observation = M @ projection((self.get_world_to_cam() @ landmarks_from_pose_hom).T).T

        # Difference between predicted and actual observation
        residual = features[:, selected_landmarks] - predicted_observation
        residual = residual.T # (N, 4)

        # Remove outliers based on residual
        chosen_landmarks = torch.abs(residual).sum(dim=1) < 20

        residual = residual[chosen_landmarks, :]
        selected_landmarks = selected_landmarks[chosen_landmarks]
        num_landmarks_seen = residual.shape[0]
        landmarks_from_pose = landmarks_from_pose[:, chosen_landmarks]
        landmarks_from_pose_hom = landmarks_from_pose_hom[:, chosen_landmarks]
        
        if num_landmarks_seen < 5:
            return

        residual = residual.flatten()

        # Jacobian of the observation with respect to the pose / measurement jacobian
        H = torch.zeros((4 * num_landmarks_seen, 3 * self.n_landmarks + 6)).double().to(self.device)
        # H = torch.zeros((4 * num_landmarks_seen, 3 * self.n_landmarks)).double().to(self.device) # - Mapping only

        for i in range(num_landmarks_seen):

            landmark_idx = selected_landmarks[i]

            P = torch.zeros((3, 4)).double().to(self.device)
            P[0, 0] = P[1, 1] = P[2, 2] = 1

            # Observation model for landmark
            out_landmark = M @ projectionJacobian((self.get_world_to_cam() @ landmarks_from_pose_hom[:, i].unsqueeze(1)).T)[0] @ self.get_world_to_cam() @ P.T
 
            # Observation model for robot pose
            out_pose = -M @ projectionJacobian((self.get_world_to_cam() @ landmarks_from_pose_hom[:, i].unsqueeze(1)).T)[0] @ circdot((self.get_world_to_cam() @ landmarks_from_pose_hom[:, i].unsqueeze(1)).T)[0]

            # print((self.get_world_to_cam() @ landmarks_from_pose_hom[:, i].unsqueeze(1)).shape)
            # print((self.get_world_to_cam() @ landmarks_from_pose_hom[:, i].unsqueeze(1))[3, :])

            # print("- ", H[4*i:4*i+4, 3*self.n_landmarks:].shape)

            H[4*i: 4*i+4, 3*landmark_idx: 3*landmark_idx+3] = out_landmark
            H[4*i: 4*i+4, 3*self.n_landmarks:] = out_pose

        I_x_V = torch.eye(4 * num_landmarks_seen) * self.measurement_noise

        print("Residual", torch.mean(torch.abs(residual)), "of ", num_landmarks_seen, "landmarks")

        # Kalman gain
        K = self.cov @ H.T @ torch.linalg.inv(H @ self.cov @ H.T + I_x_V)

        # try:
        #     KT = torch.linalg.solve(H @ self.cov @ H.T + I_x_V, H @ self.cov)
        #     K = KT.T # Shape, (3 * self.n_landmarks + 6, 4 * num_landmarks_seen)
        # except Exception as e:
        #     print("Singular matrix", e)
        #     print("Num landmarks seen: ", num_landmarks_seen)
        #     return
        # except KeyboardInterrupt:
        #     raise

        # Update covariance
        idxs_to_be_updated = torch.concatenate([
            torch.zeros(3 * selected_landmarks.shape[0]),
            torch.arange(3 * self.n_landmarks, 3 * self.n_landmarks + 6)
        ]).flatten()

        assert idxs_to_be_updated[-1] == self.cov.shape[0] - 1

        idxs_to_be_updated[0:-6:3] = selected_landmarks.flatten() * 3
        idxs_to_be_updated[1:-6:3] += idxs_to_be_updated[0:-6:3] + 1
        idxs_to_be_updated[2:-6:3] += idxs_to_be_updated[0:-6:3] + 2

        # idxs_to_be_updated[0::3] = selected_landmarks.flatten() * 3
        # idxs_to_be_updated[1::3] += idxs_to_be_updated[0::3] + 1
        # idxs_to_be_updated[2::3] += idxs_to_be_updated[0::3] + 2

        cov_to_be_updated = self.cov[idxs_to_be_updated.long(),:][:,idxs_to_be_updated.long()]

        # sub_selected_H = H[:, idxs_to_be_updated.long()]
        sub_selected_H = torch.index_select(H, 1, idxs_to_be_updated.long())

        # sub_selected_K = K[idxs_to_be_updated.long(), :]
        sub_selected_K = torch.index_select(K, 0, idxs_to_be_updated.long())

        cov_updated = (torch.eye(cov_to_be_updated.shape[0]) - sub_selected_K @ sub_selected_H) @ cov_to_be_updated

        for i, idx1 in enumerate(idxs_to_be_updated.long()):
            for j, idx2 in enumerate(idxs_to_be_updated.long()):
                self.cov[idx1, idx2] = cov_updated[i, j]

        # Update pose
        # flat_k = K[-6:, :].flatten()
        # print("Max", flat_k.max().item(), "Min", flat_k.min().item(), "Mean", flat_k.mean().item())
                
        delta = K @ residual

        # print("Delta", delta.shape)

        # print(self.exp(axangle2twist((K[-6:, :] @ residual).unsqueeze(0))[0]).numpy().round(3))
        self.mu = self.mu @ self.exp(axangle2twist((delta[-6:]).unsqueeze(0))[0])

        # Update landmarks
        self.landmarks[:, selected_landmarks] = self.landmarks[:, selected_landmarks] + delta[:-6].reshape(-1, 3).T[:, selected_landmarks]

        # print("---")

    def exp(self, x):
        return torch.linalg.matrix_exp(x).to(self.device)
    
    def get_pose(self, numpy=False):

        out = self.mu

        if numpy:
            return out.cpu().numpy()
        else:
            return out
    
    def get_cam_to_world(self):

        # 180 degree rotation around x axis
        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]).double().to(self.device)

        cam2imu = Rx @ self.cam_to_imu
        imu2world = self.get_pose()[0].double()

        return imu2world @ cam2imu
    
    def get_world_to_cam(self):

        return torch.linalg.inv(self.get_cam_to_world())

    def get_landmarks(self):
        return self.landmarks