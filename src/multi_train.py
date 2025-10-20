import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import wandb

# DDP 지원을 위한 모듈 추가
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data_processing.dataset import NGSIMDataset
from models.model_1 import DualViewInteractionAwareModel
from torch_geometric.data import Data, Batch

# --- Hyperparameters & Configuration ---
# BATCH_SIZE는 이제 GPU당 배치 크기를 의미합니다.
# 총 배치 크기 = BATCH_SIZE * GPU 수
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 1e-5
HISTORY_FRAMES = 40
FUTURE_FRAMES = 20
FEATURE_DIM = 4
LSTM_HIDDEN = 64
NUM_HEADS = 8
NUM_CLASSES = 3

# --- DDP 설정 함수 ---
def setup_ddp():
    """DDP 프로세스 그룹을 초기화하고 현재 프로세스의 순위와 장치 ID를 반환합니다."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    return rank, f'cuda:{device_id}'

def cleanup_ddp():
    """DDP 프로세스 그룹을 정리합니다."""
    dist.destroy_process_group()

def collate_fn(batch_list):
    """
    딕셔너리로 구성된 배치 리스트를 PyG의 'Batch' 객체로 변환합니다.
    DDP 환경에서도 각 프로세스가 독립적으로 이 함수를 호출하므로 수정할 필요가 없습니다.
    """
    data_list = []
    for item in batch_list:
        data = Data(
            x=item['historical_data'],
            future_x=item['future_data'],
            edge_index=item['edge_index'],
            edge_attr=item['edge_attr'],
            y=item['label'],
            num_nodes=item['historical_data'].shape[0]
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)

def train():
    # --- 1. DDP 및 장치 설정 ---
    rank, DEVICE = setup_ddp()
    is_main_process = (rank == 0)

    if is_main_process:
        print("--- Starting Model Training with DDP ---")
    
    # 스크립트 및 데이터 경로 설정
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "NGSIM", "processed_tensors")

    # --- 2. 데이터 로드 ---
    try:
        full_dataset = NGSIMDataset(data_dir=DATA_PATH)
    except FileNotFoundError as e:
        if is_main_process:
            print(f"Error loading data: {e}")
            print(f"Please ensure the processed data exists at {DATA_PATH}")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # --- DDP를 위한 DistributedSampler 설정 ---
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # DataLoader에 sampler를 전달하고, shuffle=False로 설정 (sampler가 셔플링 담당)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=val_sampler)

    if is_main_process:
        print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- 3. 클래스 가중치 계산 ---
    # 메인 프로세스에서만 계산하고 다른 프로세스에 전파
    if is_main_process:
        print("Calculating class weights for weighted loss...")
        train_labels = [sample['label'].item() for sample in train_dataset]
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        total_samples = float(sum(class_counts))
        class_weights = [total_samples / c if c > 0 else 0 for c in class_counts]
        # GPU 디바이스로 바로 텐서 이동
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    else:
        # 다른 프로세스들은 가중치를 받을 빈 텐서를 해당 GPU 디바이스에 생성
        class_weights_tensor = torch.empty(NUM_CLASSES, dtype=torch.float32).to(DEVICE)

    # 이제 모든 텐서가 GPU에 있으므로 브로드캐스트 실행
    dist.broadcast(class_weights_tensor, src=0)
    
    if is_main_process:
        print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")

    # --- 4. 모델, 옵티마이저, 손실 함수 초기화 ---
    model = DualViewInteractionAwareModel(
        feature_dim=FEATURE_DIM,
        lstm_hidden=LSTM_HIDDEN,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # DDP로 모델 감싸기
    model = DDP(model, device_ids=[int(DEVICE.split(':')[-1])])

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    if is_main_process:
        print("Model, Optimizer, and Weighted Loss function initialized for DDP.")
        wandb.init(project="lane-change-prediction", config={
            "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE * dist.get_world_size(), "epochs": EPOCHS,
            "architecture": "DDP"
        })

    # --- 5. 학습 루프 ---
    best_val_loss = float('inf')
    epochs_no_improvement = 0
    EARLY_STOPPING_PATIENCE = 30

    checkpoint_dir = os.path.join(SCRIPT_DIR, "..", "checkpoints")
    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, "best_model.pt")
        print(f"Model will be saved to {save_path}")

    for epoch in range(EPOCHS):
        # DistributedSampler가 매 에포크마다 다르게 셔플링하도록 설정
        train_sampler.set_epoch(epoch)
        
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            # DDP 모델은 내부적으로 .module을 통해 원래 모델의 forward를 호출
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward() # DDP가 여기서 그래디언트를 동기화
            optimizer.step()
            total_train_loss += loss.item()

        # --- 검증 루프 ---
        model.eval()
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch.y.size(0)
                correct_predictions += (predicted == batch.y).sum().item()

        # 각 프로세스의 결과를 집계
        # 텐서로 변환하여 all_reduce 사용
        metrics = torch.tensor([total_train_loss, total_val_loss, correct_predictions, total_samples]).to(DEVICE)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        avg_train_loss = metrics[0] / len(train_loader.dataset)
        avg_val_loss = metrics[1] / len(val_loader.dataset)
        val_accuracy = (metrics[2] / metrics[3]) * 100

        # 메인 프로세스에서만 로그 출력 및 모델 저장
        if is_main_process:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_accuracy": val_accuracy})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improvement = 0
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model... 💾")
                # DDP에서는 model.module.state_dict()를 저장
                torch.save(model.module.state_dict(), save_path)
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
            else:
                epochs_no_improvement += 1
                print(f"Validation loss did not improve. Patience: {epochs_no_improvement}/{EARLY_STOPPING_PATIENCE}")

            if epochs_no_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break
    
    # 모든 프로세스가 종료될 때까지 대기
    dist.barrier()
    if epochs_no_improvement >= EARLY_STOPPING_PATIENCE and is_main_process:
         # 다른 프로세스에도 종료 신호를 보내기 위해 에포크 루프를 중단
         # 모든 프로세스가 루프를 빠져나와 cleanup_ddp()를 호출하도록 함
         pass


    cleanup_ddp()
    if is_main_process:
        print("--- Model Training Complete ---")

if __name__ == "__main__":
    train()