import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os

# --- Constants ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate to the project root (go up 2 directories from src/data_processing)
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "NGSIM", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "NGSIM", "processed")
RAW_FILE_NAME = "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250930.csv" # Example name

# Ensure paths are absolute
RAW_DATA_DIR = os.path.abspath(RAW_DATA_DIR)
PROCESSED_DATA_DIR = os.path.abspath(PROCESSED_DATA_DIR)

# As per paper, 6s duration. Data is often at 10Hz.
HISTORY_FRAMES = 40 # 4s history
FUTURE_FRAMES = 20 # 2s future prediction window
TOTAL_FRAMES = HISTORY_FRAMES + FUTURE_FRAMES
PROXIMITY_RADIUS = 50 # Meters, for finding surrounding vehicles

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw NGSIM trajectory data from a CSV file.
    """
    print(f"Loading raw data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Raw data loaded successfully.")
    return df

def extract_trajectories(df: pd.DataFrame) -> dict:
    """
    Groups the DataFrame by Vehicle_ID to get individual vehicle trajectories.
    """
    print("Extracting individual vehicle trajectories...")
    trajectories = {}
    for vehicle_id, group in tqdm(df.groupby('Vehicle_ID')):
        trajectories[vehicle_id] = group.sort_values(by='Frame_ID').reset_index(drop=True)
    print(f"Found {len(trajectories)} unique vehicle trajectories.")
    return trajectories

def save_single_sample(sample: dict, output_dir: str, sample_idx: int):
    """
    Saves a single processed sample to disk using torch.save.
    """
    output_path = os.path.join(output_dir, f"sample_{sample_idx}.pt")
    torch.save(sample, output_path)

def segment_and_process_samples(trajectories: dict, full_df: pd.DataFrame, output_dir: str) -> int:
    """
    Iterates through trajectories, segments samples, finds surrounding vehicles,
    and saves each sample individually to disk.
    """
    print("Segmenting samples and saving individually...")
    sample_counter = 0
    lc_samples_per_vehicle = {} # MODIFIED: In-memory counter for LC samples
    
    # For faster lookups of surrounding vehicles
    frame_to_vehicles = full_df.groupby('Frame_ID')

    # Create processed directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for vehicle_id, df in tqdm(trajectories.items(), desc="Processing trajectories"):
        lc_samples_per_vehicle.setdefault(vehicle_id, 0) # MODIFIED: Initialize counter for the vehicle
        # --- 1. Lane Change Sampling ---
        lane_changes = df['Lane_ID'].diff().ne(0)
        change_indices = df[lane_changes].index
        used_indices = np.zeros(len(df), dtype=bool)

        for idx in change_indices:
            start_idx = idx - HISTORY_FRAMES
            end_idx = idx + FUTURE_FRAMES

            if start_idx >= 0 and end_idx < len(df):
                post_change_lane = df['Lane_ID'].iloc[idx]
                if (df['Lane_ID'].iloc[idx:idx+10] == post_change_lane).all() and not used_indices[start_idx:end_idx].any():
                    prev_lane = df['Lane_ID'].iloc[idx-1]
                    curr_lane = df['Lane_ID'].iloc[idx]
                    label = 2 if curr_lane > prev_lane else 1

                    start_frame = df['Frame_ID'].iloc[start_idx]
                    end_frame = df['Frame_ID'].iloc[end_idx]
                    target_trajectory = df.iloc[start_idx:end_idx]
                    
                    surrounding_trajs = find_surrounding_vehicles(
                        target_trajectory, frame_to_vehicles, start_frame, end_frame, vehicle_id
                    )

                    sample = {
                        'vehicle_id': vehicle_id,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'label': label,
                        'target_trajectory': target_trajectory,
                        'surrounding_trajectories': surrounding_trajs
                    }
                    save_single_sample(sample, output_dir, sample_counter)
                    sample_counter += 1
                    used_indices[start_idx:end_idx] = True
                    lc_samples_per_vehicle[vehicle_id] += 1 # MODIFIED: Increment counter

        # --- 2. Lane-Keeping Sampling ---
        possible_lk_starts = [
            i for i in range(len(df) - TOTAL_FRAMES) 
            if not lane_changes.iloc[i:i+TOTAL_FRAMES].any() and not used_indices[i:i+TOTAL_FRAMES].any()
        ]
        
        if possible_lk_starts:
            # MODIFIED: Get count from in-memory dict instead of slow file I/O
            num_lc_samples = lc_samples_per_vehicle[vehicle_id]
            num_lk_to_sample = min(len(possible_lk_starts), num_lc_samples)
            
            if num_lk_to_sample > 0:
                selected_starts = np.random.choice(possible_lk_starts, size=num_lk_to_sample, replace=False)

                for start_idx in selected_starts:
                    end_idx = start_idx + TOTAL_FRAMES
                    start_frame = df['Frame_ID'].iloc[start_idx]
                    end_frame = df['Frame_ID'].iloc[end_idx]
                    target_trajectory = df.iloc[start_idx:end_idx]

                    surrounding_trajs = find_surrounding_vehicles(
                        target_trajectory, frame_to_vehicles, start_frame, end_frame, vehicle_id
                    )

                    sample = {
                        'vehicle_id': vehicle_id,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'label': 0, # Lane Keep
                        'target_trajectory': target_trajectory,
                        'surrounding_trajectories': surrounding_trajs
                    }
                    save_single_sample(sample, output_dir, sample_counter)
                    sample_counter += 1

    print(f"Finished processing. Saved {sample_counter} total samples individually.")
    return sample_counter

def find_surrounding_vehicles(target_traj: pd.DataFrame, frame_to_vehicles: pd.core.groupby.DataFrameGroupBy, start_frame: int, end_frame: int, target_id: int) -> list:
    """
    Finds all surrounding vehicle trajectories for a given target vehicle's trajectory segment.
    """
    surrounding_trajectories = []
    target_start_pos = target_traj.iloc[0][['Local_X', 'Local_Y']].values.astype(np.float32)

    try:
        start_frame_data = frame_to_vehicles.get_group(start_frame)
    except KeyError:
        return []

    other_vehicles = start_frame_data[start_frame_data['Vehicle_ID'] != target_id]
    if other_vehicles.empty:
        return []

    distances = np.linalg.norm(other_vehicles[['Local_X', 'Local_Y']].values - target_start_pos, axis=1)
    proximate_vehicle_ids = other_vehicles[distances < PROXIMITY_RADIUS]['Vehicle_ID'].unique()

    if len(proximate_vehicle_ids) > 0:
        scene_df_frames = [frame_to_vehicles.get_group(f) for f in range(start_frame, end_frame) if f in frame_to_vehicles.groups]
        if not scene_df_frames:
            return []
        scene_df = pd.concat(scene_df_frames)
        
        for sv_id in proximate_vehicle_ids:
            sv_traj = scene_df[scene_df['Vehicle_ID'] == sv_id]
            if not sv_traj.empty:
                surrounding_trajectories.append(sv_traj.sort_values(by='Frame_ID').reset_index(drop=True))

    return surrounding_trajectories

if __name__ == "__main__":
    print("--- Starting Data Preprocessing for NGSIM ---")
    
    raw_data_path = os.path.join(RAW_DATA_DIR, RAW_FILE_NAME)
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw data file not found at {raw_data_path}")
    else:
        raw_df = load_raw_data(raw_data_path)
        vehicle_trajectories = extract_trajectories(raw_df)
        total_samples = segment_and_process_samples(vehicle_trajectories, raw_df, PROCESSED_DATA_DIR)
        print(f"--- Data Preprocessing complete. Total samples created: {total_samples} ---")