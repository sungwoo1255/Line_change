import torch
import os
import glob
import re
from tqdm import tqdm
import pandas as pd
import numpy as np

def sort_key(filepath):
    match = re.search(r'sample_(\d+)\.pt', os.path.basename(filepath))
    return int(match.group(1)) if match else -1

def full_preprocess_and_convert():
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the project root (go up 2 directories from src/data_processing)
    PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
    SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "NGSIM", "processed")
    TENSOR_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "NGSIM", "processed_tensors")
    
    # Ensure paths are absolute
    SOURCE_DATA_DIR = os.path.abspath(SOURCE_DATA_DIR)
    TENSOR_DATA_DIR = os.path.abspath(TENSOR_DATA_DIR)
    
    HISTORY_FRAMES = 40
    FUTURE_FRAMES = 20
    FEATURE_COLS = ['Local_X', 'Local_Y', 'v_Vel', 'v_Acc']

    if not os.path.exists(TENSOR_DATA_DIR):
        os.makedirs(TENSOR_DATA_DIR)
        print(f"Created directory: {TENSOR_DATA_DIR}")

    sample_files = glob.glob(os.path.join(SOURCE_DATA_DIR, "sample_*.pt"))
    sample_files.sort(key=sort_key)

    print(f"Found {len(sample_files)} intermediate samples to convert to tensors.")
    
    samples_processed = 0
    for sample_path in tqdm(sample_files, desc="Processing to Tensors"):
        try:
            data = torch.load(sample_path)
            
            all_vehicle_dfs = [data['target_trajectory']] + data['surrounding_trajectories']
            required_len = HISTORY_FRAMES + FUTURE_FRAMES

            valid_indices = [i for i, df in enumerate(all_vehicle_dfs) if len(df) >= required_len]

            if 0 not in valid_indices:
                continue

            valid_vehicle_dfs = [all_vehicle_dfs[i] for i in valid_indices]
            num_nodes = len(valid_vehicle_dfs)

            if num_nodes <= 1:
                continue

            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            historical_tensors = []
            future_tensors = []
            for df in valid_vehicle_dfs:
                hist_df = df.iloc[:HISTORY_FRAMES]
                future_df = df.iloc[HISTORY_FRAMES:HISTORY_FRAMES + FUTURE_FRAMES]
                historical_tensors.append(torch.tensor(hist_df[FEATURE_COLS].values, dtype=torch.float32))
                future_tensors.append(torch.tensor(future_df[FEATURE_COLS].values, dtype=torch.float32))

            historical_data = torch.stack(historical_tensors)
            future_data = torch.stack(future_tensors)

            label = torch.tensor(data['label'], dtype=torch.long)
            target_nodes = torch.tensor([0], dtype=torch.long)

            if num_nodes > 1 and edge_index.numel() > 0:
                row, col = edge_index
                last_step_features = historical_data[:, -1, :]
                source_node_features = last_step_features[row]
                dest_node_features = last_step_features[col]
                rel_pos = source_node_features[:, 0:2] - dest_node_features[:, 0:2]
                rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
                source_vel = source_node_features[:, 2].unsqueeze(-1)
                dest_vel = dest_node_features[:, 2].unsqueeze(-1)
                rel_vel = source_vel - dest_vel
                edge_attr = torch.cat([rel_dist, rel_vel], dim=-1)
            else:
                edge_attr = torch.empty((0, 2), dtype=torch.float32)

            final_tensor_dict = {
                'historical_data': historical_data,
                'future_data': future_data,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'label': label,
                'target_nodes': target_nodes
            }
            
            output_filename = os.path.basename(sample_path)
            output_path = os.path.join(TENSOR_DATA_DIR, output_filename)
            torch.save(final_tensor_dict, output_path)
            samples_processed += 1

        except Exception as e:
            # print(f"Error processing file {os.path.basename(sample_path)}. Error: {e}. Skipping.")
            pass

    print(f"\nConversion to tensors complete. {samples_processed} samples were successfully processed.")
    print(f"Final files in: {TENSOR_DATA_DIR}")

if __name__ == "__main__":
    full_preprocess_and_convert()