import torch
import torch.nn as nn
import numpy as np
import os
from MetaNet import MetaNet
from PrimitiveNet import visualize_primitives  # Assuming visualization is modularized
from SDF import determine_sphere_sdf, determine_cone_sdf
from ConeNet import visualize_cones
from spherenet import visualise_spheres
import trimesh

def test_metanet(sphere_pruning_threshold, cone_pruning_threshold, baseline_points, baseline_values, sphere_file, cone_file, sdf_file, output_dir="./output"):
    
    # Load saved sphere and cone parameters
    sdf_data = np.load(sdf_file)

    sphere_params_from_file = torch.tensor(np.load(sphere_file), dtype=torch.float32)
    cone_params_from_file = torch.tensor(np.load(cone_file), dtype=torch.float32)

    points = torch.tensor(sdf_data["sdf_points"], dtype=torch.float32)
    sdf_values = torch.tensor(sdf_data["sdf_values"], dtype=torch.float32)

    print("points shape: ", points.shape)
    print("values shape: ", sdf_values.shape)

    print("Sphere Params Shape: ", sphere_params_from_file.shape)
    print("Cone Params Shape: ", cone_params_from_file.squeeze(0).shape)

    sphere_sdf = determine_sphere_sdf(points, sphere_params_from_file)
    cone_sdf = determine_cone_sdf(points, cone_params_from_file)
    print("Sphere SDF Shape: ", sphere_sdf.shape)
    print("Cone SDF Shape: ", cone_sdf.squeeze(0).shape)

    
    sphere_errors = torch.mean((sphere_sdf - sdf_values.unsqueeze(1)) ** 2, dim=0)  # [num_spheres]
    cone_errors = torch.mean((cone_sdf.squeeze(0) - sdf_values.unsqueeze(1)) ** 2, dim=0)  # [num_cones]

    sphere_paramms_with_errors = torch.cat((sphere_params_from_file, sphere_errors.unsqueeze(1)), dim=1)
    cone_paramms_with_errors = torch.cat((cone_params_from_file.squeeze(0), cone_errors.unsqueeze(1)), dim=1)

    print("Sphere Params with Errors Shape: ", sphere_paramms_with_errors.shape)
    print("Cone Params with Errors Shape: ", cone_paramms_with_errors.shape)

    #print max and min values for errors
    print("     Sphere Errors Max: ", sphere_errors.max().item())
    print("     Sphere Errors Min: ", sphere_errors.min().item())
    print("     Cone Errors Max: ", cone_errors.max().item())
    print("     Cone Errors Min: ", cone_errors.min().item())

    sphere_mask = sphere_paramms_with_errors[:, -1] < sphere_pruning_threshold
    cone_mask = cone_paramms_with_errors[:, -1] < cone_pruning_threshold

    pruned_sphere_params = sphere_paramms_with_errors[sphere_mask]
    pruned_cone_params = cone_paramms_with_errors[cone_mask]

    pruned_sphere_params = pruned_sphere_params[:, :-1]
    pruned_cone_params = pruned_cone_params[:, :-1]

    print("Pruned Sphere Params Shape: ", pruned_sphere_params.shape)
    print("Pruned Cone Params Shape: ", pruned_cone_params.shape)

    # visualise_spheres(points, sdf_values, sphere_params_from_file, reference_model=None, save_path=None)
    # visualise_spheres(points, sdf_values, pruned_sphere_params, reference_model=None, save_path=None)

    print("Cone Params Shape: ", cone_params_from_file.shape)

    # visualize_cones(points, sdf_values, cone_params_from_file, save_path=None)
    # visualize_cones(points, sdf_values, pruned_cone_params, save_path=None)

    visualize_primitives(baseline_points, baseline_values, pruned_sphere_params, pruned_cone_params)

if __name__ == "__main__":

    dataset_path = "./reference_models_processed"
    name = "dog"

    # Example paths for input files
    sphere_file = f"./output/{name}_sphere_params.npy"
    cone_file = f"./output/{name}_cone_params.npy"
    sdf_file = f"./reference_models_processed/{name}/voxel_and_sdf.npz"

    data = np.load(os.path.join(dataset_path, name, "voxel_and_sdf.npz"))

    voxel_data = data["voxels"]
    points = data["sdf_points"]
    values = data["sdf_values"]

    sphere_pruning_threshold =0.35
    cone_pruning_threshold = 0.5


    test_metanet(sphere_pruning_threshold, cone_pruning_threshold, points, values, sphere_file, cone_file, sdf_file)


