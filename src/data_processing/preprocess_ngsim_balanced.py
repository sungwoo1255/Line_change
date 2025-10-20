import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os

## 해당 파일이 데이터 발란스에 맞게해

# --- Constants ---
RAW_DATA_DIR = "/home/sean/vla/lane_change_prediction/data/NGSIM/raw"
PROCESSED_DATA_DIR = "/home/sean/vla/lane_change_prediction/data/NGSIM/processed_balanced"
RAW_FILE_NAME = "Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250930.csv"

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

def segment_and_process_globally_balanced(trajectories: dict, full_df: pd.DataFrame, output_dir: str) -> int:
    """
    Performs a two-pass process to create a globally balanced dataset.
    Pass 1: Identify all lane change (LC) samples and all possible lane keep (LK) candidates.
    Pass 2: Downsample LK candidates to match the number of LC samples and save everyone.
    """
    print("Segmenting samples with GLOBAL balancing strategy...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    lc_samples_info = []
    lk_candidate_info = []
    
    frame_to_vehicles = full_df.groupby('Frame_ID')

    # --- Pass 1: Identify all potential samples ---
    print("Pass 1: Identifying all Lane Change samples and Lane Keep candidates...")
    for vehicle_id, df in tqdm(trajectories.items(), desc="Identifying samples"):
        lane_changes = df['Lane_ID'].diff().ne(0)
        change_indices = df[lane_changes].index
        
        # Store used indices for this vehicle to prevent overlap
        used_indices = np.zeros(len(df), dtype=bool)

        # Find LC samples
        for idx in change_indices:
            start_idx = idx - HISTORY_FRAMES
            end_idx = idx + FUTURE_FRAMES
            if start_idx >= 0 and end_idx < len(df) and not used_indices[start_idx:end_idx].any():
                post_change_lane = df['Lane_ID'].iloc[idx]
                if (df['Lane_ID'].iloc[idx:idx+10] == post_change_lane).all():
                    prev_lane = df['Lane_ID'].iloc[idx-1]
                    label = 2 if df['Lane_ID'].iloc[idx] > prev_lane else 1
                    lc_samples_info.append({'vehicle_id': vehicle_id, 'start_idx': start_idx, 'label': label})
                    used_indices[start_idx:end_idx] = True

        # Find LK candidates
        for i in range(len(df) - TOTAL_FRAMES):
            if not used_indices[i:i+TOTAL_FRAMES].any():
                # Check if there is any lane change in this window
                if not df['Lane_ID'].iloc[i:i+TOTAL_FRAMES].nunique() > 1:
                    lk_candidate_info.append({'vehicle_id': vehicle_id, 'start_idx': i})
                    # Mark as used to avoid creating overlapping LK samples
                    used_indices[i:i+TOTAL_FRAMES] = True

    # --- Pass 2: Balance, process, and save ---
    print(f"Found {len(lc_samples_info)} Lane Change samples.")
    print(f"Found {len(lk_candidate_info)} potential Lane Keep samples.")

    # Down-sample LK candidates to match the number of LC samples (1:1 ratio)
    num_lk_to_sample = len(lc_samples_info)
    if len(lk_candidate_info) < num_lk_to_sample:
        print(f"Warning: Not enough unique Lane Keep segments. Using all {len(lk_candidate_info)} available.")
        num_lk_to_sample = len(lk_candidate_info)
        
    selected_lk_indices = np.random.choice(len(lk_candidate_info), size=num_lk_to_sample, replace=False)
    selected_lk_samples = [lk_candidate_info[i] for i in selected_lk_indices]
    
    # Combine and shuffle
    all_samples_to_process = lc_samples_info + selected_lk_samples
    np.random.shuffle(all_samples_to_process)
    
    print(f"Balanced sampling: Using {len(lc_samples_info)} LC samples and {len(selected_lk_samples)} LK samples.")
    
    sample_counter = 0
    print("Pass 2: Processing and saving balanced dataset...")
    for sample_info in tqdm(all_samples_to_process, desc="Saving balanced samples"):
        vehicle_id = sample_info['vehicle_id']
        start_idx = sample_info['start_idx']
        label = sample_info.get('label', 0) # Default to 0 (LK) if not specified
        
        df = trajectories[vehicle_id]
        end_idx = start_idx + TOTAL_FRAMES
        
        start_frame = df['Frame_ID'].iloc[start_idx]
        end_frame = df['Frame_ID'].iloc[end_idx]
        target_trajectory = df.iloc[start_idx:end_idx]

        surrounding_trajs = find_surrounding_vehicles(
            target_trajectory, frame_to_vehicles, start_frame, end_frame, vehicle_id
        )

        sample_to_save = {
            'vehicle_id': vehicle_id,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'label': label,
            'target_trajectory': target_trajectory,
            'surrounding_trajectories': surrounding_trajs
        }
        save_single_sample(sample_to_save, output_dir, sample_counter)
        sample_counter += 1
        
    print(f"Finished processing. Saved {sample_counter} total balanced samples individually.")
    return sample_counter

if __name__ == "__main__":
    print("--- Starting Data Preprocessing for NGSIM (Globally Balanced) ---")
    
    raw_data_path = os.path.join(RAW_DATA_DIR, RAW_FILE_NAME)
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw data file not found at {raw_data_path}")
    else:
        raw_df = load_raw_data(raw_data_path)
        vehicle_trajectories = extract_trajectories(raw_df)
        # Call the new globally balanced function
        total_samples = segment_and_process_globally_balanced(vehicle_trajectories, raw_df, PROCESSED_DATA_DIR)
        print(f"--- Data Preprocessing complete. Total samples created: {total_samples} ---")