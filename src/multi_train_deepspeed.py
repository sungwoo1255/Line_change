import torch
import torch.nn as nn
import os
import numpy as np
import wandb
import argparse
import deepspeed

# DDP와 달리 DeepSpeed는 자체적으로 분산 환경을 초기화합니다.
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_processing.dataset import NGSIMDataset
from models.model_1 import DualViewInteractionAwareModel
from torch_geometric.data import Data, Batch

# --- Hyperparameters & Configuration ---
# DeepSpeed 설정에서 배치 크기를 관리하므로, 여기서는 마이크로 배치 크기를 정의합니다.
MICRO_BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 1e-5
HISTORY_FRAMES = 40
FUTURE_FRAMES = 20
FEATURE_DIM = 4
LSTM_HIDDEN = 64
NUM_HEADS = 8
NUM_CLASSES = 3

def collate_fn(batch_list):
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
    # --- 1. Argument Parser 및 DeepSpeed 초기화 ---
    parser = argparse.ArgumentParser(description='DeepSpeed Lane Change Prediction Training')
    # DeepSpeed launcher가 전달하는 local_rank 인자를 받도록 추가합니다.
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # DeepSpeed 설정 관련 인자들을 추가합니다.
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # DeepSpeed 분산 환경 초기화
    deepspeed.init_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])
    DEVICE = f'cuda:{local_rank}'
    torch.cuda.set_device(DEVICE)
    
    is_main_process = (dist.get_rank() == 0)

    if is_main_process:
        print("--- Starting Model Training with DeepSpeed ---")

    # 스크립트 및 데이터 경로 설정
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "NGSIM", "processed_tensors")

    # --- 2. 데이터 로드 ---
    full_dataset = NGSIMDataset(data_dir=DATA_PATH)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=val_sampler)

    if is_main_process:
        print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- 3. 클래스 가중치 계산 ---
    if is_main_process:
        train_labels = [sample['label'].item() for sample in train_dataset]
        class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        class_weights = [sum(class_counts) / c if c > 0 else 0 for c in class_counts]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    else:
        class_weights_tensor = torch.empty(NUM_CLASSES, dtype=torch.float32).to(DEVICE)
    
    dist.broadcast(class_weights_tensor, src=0)
    if is_main_process:
        print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")

    # --- 4. 모델, 손실 함수 초기화 ---
    model = DualViewInteractionAwareModel(
        feature_dim=FEATURE_DIM, lstm_hidden=LSTM_HIDDEN, num_heads=NUM_HEADS, num_classes=NUM_CLASSES
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.half())

    # --- 5. DeepSpeed 엔진 초기화 ---
    # DeepSpeed는 자체적으로 모델, 옵티마이저, 스케줄러를 래핑합니다.
    # 옵티마이저는 DeepSpeed 설정 파일/딕셔너리를 통해 설정됩니다.
    model_engine, _, _, _ = deepspeed.initialize(
        args=args, 
        model=model, 
        model_parameters=model.parameters()
    )

    if is_main_process:
        print("DeepSpeed engine initialized.")
        wandb.init(project="lane-change-prediction-deepspeed", config={
            "learning_rate": LEARNING_RATE, 
            "micro_batch_size": MICRO_BATCH_SIZE, 
            "global_batch_size": MICRO_BATCH_SIZE * dist.get_world_size(),
            "epochs": EPOCHS,
            "architecture": "DeepSpeed"
        })

    # --- 6. 학습 루프 ---
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(SCRIPT_DIR, "..", "checkpoints_deepspeed")

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        
        model_engine.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(model_engine.device)
            
            # FP16 학습을 위해 입력 데이터 타입을 Half로 변환
            batch.x = batch.x.half()
            batch.future_x = batch.future_x.half()
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr.half()

            outputs = model_engine(batch)
            loss = criterion(outputs, batch.y)
            
            # DeepSpeed의 backward와 step 사용
            model_engine.backward(loss)
            model_engine.step()
            
            total_train_loss += loss.item()

        # --- 검증 루프 ---
        model_engine.eval()
        total_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(model_engine.device)

                # FP16 학습을 위해 입력 데이터 타입을 Half로 변환
                batch.x = batch.x.half()
                batch.future_x = batch.future_x.half()
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    batch.edge_attr = batch.edge_attr.half()

                outputs = model_engine(batch)
                loss = criterion(outputs, batch.y)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch.y.size(0)
                correct_predictions += (predicted == batch.y).sum().item()

        metrics = torch.tensor([total_train_loss, total_val_loss, correct_predictions, total_samples]).to(model_engine.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        avg_train_loss = metrics[0] / len(train_loader.dataset)
        avg_val_loss = metrics[1] / len(val_loader.dataset)
        val_accuracy = (metrics[2] / metrics[3]) * 100

        if is_main_process:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_accuracy": val_accuracy})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # DeepSpeed 체크포인트 저장
                model_engine.save_checkpoint(checkpoint_dir, tag=f'best_model')
                print(f"New best validation loss: {best_val_loss:.4f}. Saving DeepSpeed checkpoint... 💾")

    if is_main_process:
        print("--- Model Training Complete ---")

if __name__ == "__main__":
    train()
