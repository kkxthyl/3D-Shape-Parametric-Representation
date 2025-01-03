import torch
import torch.nn as nn
import numpy as np
import trimesh
from dgcnn import DGCNNFeat
from Decoder import Decoder
from SDF import determine_cylinder_sdf

class CylinderNet(nn.Module):
    def __init__(self, num_cylinders=32):
        super(CylinderNet, self).__init__()
        self.num_cylinders = num_cylinders
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()
        self.feature_mapper = nn.Linear(512, num_cylinders * 8)

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        # Decode the features into cylinder parameters
        cylinder_params = self.decoder(features)

        cylinder_params = self.feature_mapper(cylinder_params).view(self.num_cylinders, 8)
        cylinder_params = torch.sigmoid(cylinder_params.view(-1, 8))

        cylinder_adder = torch.tensor([-0.8, -0.8, -0.8, -1.0, -1.0, -1.0, 0.1, 0.1]).to(cylinder_params.device)
        cylinder_multiplier = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.1, 0.1]).to(cylinder_params.device)
        cylinder_params = cylinder_params * cylinder_multiplier + cylinder_adder

        cylinder_sdf = determine_cylinder_sdf(query_points, cylinder_params)

        return cylinder_sdf, cylinder_params
    

def visualize_cylinders(cylinder_params, points, values, reference_model=None, save_path=None):
    cylinder_params = cylinder_params.cpu().detach().numpy()
    cylinder_centers = cylinder_params[:, :3]
    cylinder_axes = cylinder_params[:, 3:6]
    cylinder_radii = cylinder_params[:, 6]
    cylinder_heights = cylinder_params[:, 7]
    scene = trimesh.Scene()

    for i in range(cylinder_params.shape[0]):
        cyl = trimesh.creation.cylinder(cylinder_heights[i], cylinder_radii[i])

        # Normalize the axis vector
        axis = cylinder_axes[i]
        if np.linalg.norm(axis) < 1e-6:
            axis = np.array([0, 0, 1])  
        axis = axis / np.linalg.norm(axis)

        # rotation matrix with the orientation vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, axis)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_matrix = np.eye(4)  # identity matrix
        else:
            rotation_angle = np.arccos(np.dot(z_axis, axis))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

        cyl.apply_transform(rotation_matrix)
        cyl.apply_translation(cylinder_centers[i])
        scene.add_geometry(cyl)
    
    inside_points = points[values < 0]
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices)) 
        scene.add_geometry(inside_points)
        
    if save_path is not None:
        scene.export(save_path)

    scene.show()
