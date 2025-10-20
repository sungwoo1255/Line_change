import torch
from torch.utils.data import Dataset
import os
import glob
import re

def sort_key(filepath):
    """Extracts the number from a filename like 'sample_123.pt' for sorting."""
    match = re.search(r'sample_(\d+)\.pt', os.path.basename(filepath))
    return int(match.group(1)) if match else -1

class NGSIMDataset(Dataset):
    """
    PyTorch Dataset class for loading FULLY PREPROCESSED NGSIM tensors.
    This class loads tensor files and ensures they are valid 
    (e.g., no 'None' values) before returning them.
    """
    def __init__(self, data_dir: str, **kwargs):
        """
        Args:
            data_dir: Path to the directory containing final tensor files (sample_*.pt).
        """
        super().__init__()
        print(f"Scanning for final tensor files in {data_dir}...")
        self.sample_files = glob.glob(os.path.join(data_dir, "sample_*.pt"))
        if not self.sample_files:
            raise FileNotFoundError(f"No tensor sample files found in {data_dir}.")
        
        self.sample_files.sort(key=sort_key)
        
        # [수정] 엣지 속성의 차원을 정의합니다. (train.py의 collate_fn과 일치해야 함)
        # 예: (상대 거리 X, 상대 거리 Y) -> 2
        self.edge_attr_dim = 2 
        
        print(f"Dataset initialized. Found {len(self.sample_files)} preprocessed tensor samples.")

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        """
        Loads a sample and validates its graph components (edge_index, edge_attr).
        """
        sample_path = self.sample_files[idx]
        sample = torch.load(sample_path)

        # --- [수정] 데이터 유효성 검사 및 수정 ---
        
        # 1. edge_index가 없는 경우 (None이거나 키가 없음)
        if sample.get('edge_index') is None:
            # 빈 [2, 0] 텐서로 교체
            sample['edge_index'] = torch.empty((2, 0), dtype=torch.long)

        # 2. edge_attr가 없는 경우 (None이거나 키가 없음)
        if sample.get('edge_attr') is None:
            num_edges = sample['edge_index'].shape[1]
            
            if num_edges == 0:
                # 엣지가 없으므로, [0, D_attr] 모양의 빈 텐서로 교체
                sample['edge_attr'] = torch.empty((0, self.edge_attr_dim), dtype=torch.float32)
            else:
                # [문제 상황] 엣지는 있으나 속성이 None인 경우
                # 0으로 채워진 더미 텐서를 생성하고 경고 출력
                print(f"Warning: sample {os.path.basename(sample_path)} has {num_edges} edges but 'edge_attr' is None. Creating zero-filled tensor.")
                sample['edge_attr'] = torch.zeros((num_edges, self.edge_attr_dim), dtype=torch.float32)

        return sample

if __name__ == '__main__':
    # --- Example of how to use the simplified Dataset ---
    # 이 스크립트 파일 위치 기준으로 경로 수정
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/NGSIM/processed_tensors"))

    try:
        dataset = NGSIMDataset(data_dir=data_dir)
        print(f"Successfully created dataset with {len(dataset)} samples.")
        
        # Get one sample to inspect
        sample_data = dataset[0]
        print("\n--- Inspecting a single preprocessed tensor sample (after validation) ---")
        print(f"Label: {sample_data['label']}")
        print(f"Historical Data Shape: {sample_data['historical_data'].shape}")
        print(f"Edge Index Shape: {sample_data['edge_index'].shape}")
        # [수정] edge_attr shape도 출력
        print(f"Edge Attr Shape: {sample_data['edge_attr'].shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")