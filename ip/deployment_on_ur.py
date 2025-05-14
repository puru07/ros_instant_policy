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
    
    # Create demo dictionary
    demo = {
        'pcds': pcds,
        'T_w_es': transforms['matrices'],
        'grips': np.zeros(len(pcds))  # Assuming gripper state is not recorded
    }
    print('T_w_es: ', demo['T_w_es'])
    
    return demo

if __name__ == '__main__':
    ####################################################################################################################
    # Define rollout parameters. 
    num_demos = 1 # originally 2
    num_traj_wp = 10
    num_diffusion_iters = 4
    compile_models = False
    max_execution_steps = 100
    ####################################################################################################################
    # Load and prepare trained model.

    checkpoint_default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model_path = checkpoint_default_dir
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
    config['num_layers'] = 2

    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = num_demos
    config['num_diffusion_iters_test'] = num_diffusion_iters

    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=False,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=max(num_demos, 1))
    model.eval()

    if compile_models:
        model.model.compile_models()
    ####################################################################################################################
    # Process demonstrations.
    # Load demonstrations from assets/rs_data folder
    demo_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'rs_data')
    
    # Get all demo folders and sort by timestamp (folder name) in descending order
    demo_folders = glob.glob(os.path.join(demo_base_dir, '*'))
    demo_folders.sort(key=lambda x: os.path.basename(x), reverse=True)  # Sort by folder name (timestamp) in descending order
    
    # Take the most recent num_demos folders
    demo_folders = demo_folders[:num_demos]
    
    print(f"Loading {len(demo_folders)} most recent demonstrations:")
    for folder in demo_folders:
        print(f"  - {os.path.basename(folder)}")
    
    demos = []
    for folder in demo_folders:
        demo = load_demo_from_folder(folder)
        demos.append(demo)

    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }
    for i, demo in enumerate(demos):
        full_sample['demos'][i] = sample_to_cond_demo(demo, num_traj_wp)
        num_traj_wp = len(full_sample['demos'][i]['obs'])
        # assert len(full_sample['demos'][i]['obs']) == num_traj_wp
    ####################################################################################################################

    # Rollout the model.
    for k in range(max_execution_steps):
        T_w_e = None  # TODO: end-effector pose in the world frame, [4, 4].
        pcd_w = None  # TODO: segmented point cloud observation in the world frame, [N, 3].
        grip = None  # TODO: whether the gripper is closed or opened, [0, 1]
        full_sample['live']['obs'] = [transform_pcd(subsample_pcd(pcd_w), np.linalg.inv(T_w_e))]
        full_sample['live']['grips'] = [grip]
        full_sample['live']['actions_grip'] = [np.zeros(8)]
        full_sample['live']['T_w_es'] = [T_w_e]
        full_sample['live']['actions'] = [T_w_e.reshape(1, 4, 4).repeat(config['pre_horizon'], axis=0)]
        data = save_sample(full_sample, None)
        
        # For efficiency, pre-compute and cache geometry embeddings for the demos. 
        if k == 0:
            demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                data.to(model.config['device']))
        data.live_scene_node_embds, data.live_scene_node_pos =\
            model.model.get_live_scene_emb(data.to(model.config['device']))
        data.demo_scene_node_embds = demo_scene_node_embds.clone()
        data.demo_scene_node_pos = demo_scene_node_pos.clone()
        
        # Inference on the model.
        with torch.no_grad():
            with torch.autocast(dtype=torch.float32, device_type=model.config['device']):
                actions, grips = model.test_step(data.to(model.config['device']), 0)
            actions = actions.squeeze().cpu().numpy()
            grips = grips.squeeze().cpu().numpy()
        
        # TODO: Use whatever controller you have to execute all or part of the predicted actions.
        # TODO: actions: [Pred_horizon, 4, 4] are relative transforms of the end-effector.
        # TODO: To get next pose of the end-effector in the world frame you use T_w_e @ actions[j].
        # TODO: grips: [Pred_horizon, 1] are open and close commands: -1 is close, 1 is open.
