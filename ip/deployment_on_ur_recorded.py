#!/usr/bin/env python3
'''
This scripts shows and example of how Instant Policy could be used at deployment.
'''

import sys
sys.path.insert(0, '/home/mcqueen/anaconda3/envs/ip_env/lib/python3.10/site-packages')  # Adjust this path

import pickle
from ip.models.diffusion import GraphDiffusion
from ip.utils.data_proc import *
import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import torch

def load_live_data_from_folder(folder_path):
    """Load live data from a folder containing cropped PCD files and transformation matrices."""
    # Load transformation matrices
    transform_file = os.path.join(folder_path, 'tool0_transforms.pkl')
    with open(transform_file, 'rb') as f:
        transforms = pickle.load(f)
    
    # Load cropped PCD files
    pcd_dir = os.path.join(folder_path, 'cropped_pcds')
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    
    # Create a list of live data dictionaries
    live_data_list = []
    for i, pcd_file in enumerate(pcd_files):
        # Read PCD file
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
            # Skip header
            points = []
            for line in lines[11:]:  # Skip PCD header
                x, y, z = map(float, line.strip().split())
                points.append([x, y, z])
            pcd = np.array(points, dtype=np.float64)  # Ensure float64 type
        
        # Create live data dictionary for this point cloud
        live_data = {
            'pcds': [pcd],  # Single point cloud in a list
            'T_w_es': [transforms['matrices'][i]],  # Single transform in a list
            'grips': np.array([0.0])  # Single grip state
        }
        live_data_list.append(live_data)
    
    # Debug: Print live data information
    print(f"Loaded live data from {folder_path}:")
    print(f"  Number of live data entries: {len(live_data_list)}")
    print(f"  Live data keys: {live_data_list[0].keys()}")
    if len(live_data_list) > 0:
        print(f"  Point cloud shape: {live_data_list[0]['pcds'][0].shape}")
        print(f"  Point cloud dtype: {live_data_list[0]['pcds'][0].dtype}")
    
    return live_data_list

def load_demo_from_folder(folder_path):
    """Load demonstration data from a folder containing cropped PCD files and transformation matrices."""
    # Load transformation matrices
    transform_file = os.path.join(folder_path, 'tool0_transforms.pkl')
    with open(transform_file, 'rb') as f:
        transforms = pickle.load(f)
    
    # Load cropped PCD files
    pcd_dir = os.path.join(folder_path, 'cropped_pcds')
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, '*.pcd')))
    
    # Read PCD files
    pcds = []
    for pcd_file in pcd_files:
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
            # Skip header
            points = []
            for line in lines[11:]:  # Skip PCD header
                x, y, z = map(float, line.strip().split())
                points.append([x, y, z])
            pcds.append(np.array(points))
    
    # Trim transformations to match number of PCDs
    num_pcds = len(pcds)
    transforms['matrices'] = transforms['matrices'][:num_pcds]
    
    # Create a single demo dictionary containing all PCDs and transforms
    demo_sample = {
        'pcds': pcds,  # List of all point clouds
        'T_w_es': transforms['matrices'],  # List of all transforms
        'grips': np.zeros(len(pcds))  # Gripper states for all points
    }
    
    # Debug: Print demo_sample information
    print(f"Loaded demo_sample from {folder_path}:")
    print(f"  Number of PCDs: {len(pcds)}")
    print(f"  Number of transforms: {len(transforms['matrices'])}")
    print(f"  Demo_sample keys: {demo_sample.keys()}")
    if len(pcds) > 0:
        print(f"  First PCD shape: {pcds[0].shape}")
        print(f"  First transform shape: {transforms['matrices'][0].shape}")
    
    return demo_sample

def process_data(model, demo, live_data_list):
    """Process the loaded data through the model."""
    try:
        # Rename 'pcds' to 'obs' in demo data
        demo['obs'] = demo.pop('pcds')
        full_sample = {
            'demos': [demo],
            'live': dict(),
        }
        
        # Print shapes of demo data
        print("\nDemo data shapes:")
        for key, value in full_sample['demos'][0].items():
            if isinstance(value, list):
                print(f"{key}: list of length {len(value)}")
                if len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        print(f"  - First element shape: {value[0].shape}")
            elif isinstance(value, np.ndarray):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        
        # Process each live data entry
        for i, live_data in enumerate(live_data_list):
            print(f"\nProcessing live data entry {i}")
            
            # Prepare data for model rollout


            T_w_e = live_data['T_w_es'][0]
            # Extract the point cloud from the list and ensure it's a Nx3 array
            pcd = live_data['pcds'][0] if isinstance(live_data['pcds'], list) else live_data['pcds']

            full_sample['live']['obs'] = [transform_pcd(subsample_pcd(pcd), np.linalg.inv(T_w_e))]
            full_sample['live']['grips'] = live_data['grips']
            full_sample['live']['actions_grip'] = [np.zeros(8)]
            full_sample['live']['T_w_es'] = [T_w_e]
            full_sample['live']['actions'] = [full_sample['live']['T_w_es'][0].reshape(1, 4, 4).repeat(8, axis=0)]
            
            
            print(f"\nLive Data {i} shapes:")
            print("obs shape:", np.array(full_sample['live']['obs']).shape)
            print("grips shape:", np.array(full_sample['live']['grips']).shape)
            print("actions_grip shape:", np.array(full_sample['live']['actions_grip']).shape)
            print("T_w_es shape:", np.array(full_sample['live']['T_w_es']).shape)
            print("actions shape:", np.array(full_sample['live']['actions']).shape)
            
            
            # Convert to model input format
            data = save_sample(full_sample, None)
            
            # Print shapes after save_sample
            print("\nData shapes after save_sample:")
            if hasattr(data, 'demo_scene_node_embds'):
                print("Demo scene node embeddings shape:", data.demo_scene_node_embds.shape)
            if hasattr(data, 'live_scene_node_embds'):
                print("Live scene node embeddings shape:", data.live_scene_node_embds.shape)
            if hasattr(data, 'demo_scene_node_pos'):
                print("Demo scene node positions shape:", data.demo_scene_node_pos.shape)
            if hasattr(data, 'live_scene_node_pos'):
                print("Live scene node positions shape:", data.live_scene_node_pos.shape)
            
            # For efficiency, pre-compute and cache geometry embeddings for the demos
            demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                data.to(model.config['device']))
            
            # Get live scene embeddings
            data.live_scene_node_embds, data.live_scene_node_pos =\
                model.model.get_live_scene_emb(data.to(model.config['device']))
            data.demo_scene_node_embds = demo_scene_node_embds.clone()
            data.demo_scene_node_pos = demo_scene_node_pos.clone()
            
            # Print shapes after getting embeddings
            print("\nData shapes after getting embeddings:")
            print("Demo scene node embeddings shape:", data.demo_scene_node_embds.shape)
            print("Live scene node embeddings shape:", data.live_scene_node_embds.shape)
            print("Demo scene node positions shape:", data.demo_scene_node_pos.shape)
            print("Live scene node positions shape:", data.live_scene_node_pos.shape)
            
            # Inference on the model
            with torch.no_grad():
                with torch.autocast(dtype=torch.float32, device_type=model.config['device']):
                    actions, grips = model.test_step(data.to(model.config['device']), 0)
                actions = actions.squeeze().cpu().numpy()
                grips = grips.squeeze().cpu().numpy()
            
            print("Predicted actions:", actions.shape)
            print(actions)
            print("Predicted grips:", grips.shape)
            
            return actions, grips

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

def main():
    ####################################################################################################################
    # Define rollout parameters. 
    num_demos = 1
    num_traj_wp = 10
    num_diffusion_iters = 4
    compile_models = False
    ####################################################################################################################
    # Load and prepare trained model.
    checkpoint_default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model_path = './checkpoints'
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
    

    
    config['num_layers'] = 2
    config['device'] = 'cpu'
    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = num_demos
    config['num_diffusion_iters_test'] = num_diffusion_iters

    # Print all config values
    print("\nConfiguration values:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=False,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=max(num_demos, 1))
    model.eval()

    if compile_models:
        model.model.compile_models()
    ####################################################################################################################
    # Process demonstrations.
    # Load demonstrations from assets/rs_data folder
    demo_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ip','assets', 'rs_data')
    print(f"Looking for demo folders in: {demo_base_dir}")
    
    # Check if directory exists
    if not os.path.exists(demo_base_dir):
        raise ValueError(f"Demo directory does not exist: {demo_base_dir}")
    
    # Get all demo folders and sort by timestamp (folder name) in descending order
    demo_folders = glob.glob(os.path.join(demo_base_dir, '*'))
    print(f"Found {len(demo_folders)} folders in demo directory")
    
    if not demo_folders:
        print("\nNo demo folders found. Please ensure:")
        print("1. The directory exists: assets/rs_data")
        print("2. The directory contains at least 2 demo folders")
        print("3. Each demo folder should contain:")
        print("   - tool0_transforms.pkl")
        print("   - cropped_pcds/ directory with .pcd files")
        raise ValueError("No demo folders found in the specified directory")
    
    demo_folders.sort(key=lambda x: os.path.basename(x), reverse=True)  # Sort by folder name (timestamp) in descending order
    print("\nFound demo folders:")

    if len(demo_folders) < 2:
        print("\nNeed at least 2 demo folders:")
        print("1. One for demonstration data")
        print("2. One for live data")
        raise ValueError(f"Found only {len(demo_folders)} demo folder(s), need at least 2")
    
    # Use the most recent folder for demo
    demo_folder = demo_folders[0]
    # Use the second most recent folder for live data
    # live_data_folder = demo_folders[1]
    live_data_folder = demo_folders[0]
    
    print(f"\nUsing demo from: {os.path.basename(demo_folder)}")
    print(f"Using live data from: {os.path.basename(live_data_folder)}")
    
    # Load demo data
    demo_sample = load_demo_from_folder(demo_folder)
    live_data_list = load_live_data_from_folder(live_data_folder)
    
    # Process the data through the model
    actions, grips = process_data(model, demo_sample, live_data_list)

if __name__ == '__main__':
    main()
