import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import numpy as np
import wandb

from data_processing.dataset import NGSIMDataset
from models.model import DualViewInteractionAwareModel


# --- Hyperparameters & Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get the absolute path to the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the data directory
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "NGSIM", "processed_tensors")
HISTORY_FRAMES = 40
FUTURE_FRAMES = 20

# Model dimensions
FEATURE_DIM = 4 # Local_X, Local_Y, v_Vel, v_Acc
LSTM_HIDDEN = 64
NUM_HEADS = 8
NUM_CLASSES = 3 # 0: Keep, 1: Right, 2: Left

# Training parameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
EPOCHS = 500

def collate_fn(batch):
    """
    Custom collate function to handle batches of graph data with varying sizes.
    """
    hist_data_list, future_data_list, edge_index_list, edge_attr_list, labels_list = [], [], [], [], []
    target_node_indices = []
    node_offset = 0

    for item in batch:
        hist_data = item['historical_data']
        future_data = item['future_data']
        edge_index = item['edge_index']
        # Use .get() for safety, in case a sample is missing the attribute
        edge_attr = item.get('edge_attr') 
        label = item['label']
        num_nodes = hist_data.shape[0]

        hist_data_list.append(hist_data)
        future_data_list.append(future_data)
        labels_list.append(label)

        if edge_attr is not None:
            edge_attr_list.append(edge_attr)

        # Add offset to edge_index to create a single large graph
        if num_nodes > 1:
            edge_index_list.append(edge_index + node_offset)
        
        # The target vehicle is always the first node in each sample
        target_node_indices.append(node_offset)
        node_offset += num_nodes

    # Combine all data into batch tensors
    batch_hist_data = torch.cat(hist_data_list, dim=0)
    batch_future_data = torch.cat(future_data_list, dim=0)
    batch_edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    # Assuming edge_attr has 2 features. Create empty tensor if no edge_attr is found.
    batch_edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, 2))
    batch_labels = torch.stack(labels_list, dim=0)
    batch_target_nodes = torch.tensor(target_node_indices, dtype=torch.long)

    return {
        "historical_data": batch_hist_data,
        "future_data": batch_future_data,
        "edge_index": batch_edge_index,
        "edge_attr": batch_edge_attr,
        "labels": batch_labels,
        "target_nodes": batch_target_nodes
    }

def train():
    print("--- Starting Model Training ---")
    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    try:
        full_dataset = NGSIMDataset(data_dir=DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure the processed data exists at {DATA_PATH}")
        return

    # Splitting dataset (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- 2. Calculate Class Weights for Imbalanced Dataset ---
    print("Calculating class weights for weighted loss...")
    # Efficiently get all labels from the training set
    train_labels = [sample['label'].item() for sample in train_dataset]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    
    # Calculate weights as the inverse of class frequency
    total_samples = float(sum(class_counts))
    class_weights = [total_samples / c if c > 0 else 0 for c in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")

    # --- 3. Initialize Model, Optimizer, Loss ---
    model = DualViewInteractionAwareModel(
        feature_dim=FEATURE_DIM,
        lstm_hidden=LSTM_HIDDEN,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use the calculated weights in the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("Model, Optimizer, and Weighted Loss function initialized.")

    wandb.init(project="lane-change-prediction", 
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lstm_hidden": LSTM_HIDDEN,
            "num_heads": NUM_HEADS,
            "feature_dim": FEATURE_DIM
        })

    # --- 4. Training Loop ---

    # [추가] 최적 모델 저장 및 조기 종료를 위한 변수 초기화
    best_val_loss = float('inf')
    epochs_no_improvement = 0
    EARLY_STOPPING_PATIENCE = 30

    # [추가/수정] 모델 저장 경로 (절대 경로 사용)
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pt")
    print(f"Model will be saved to {save_path}")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            # Move data to device
            hist_data = batch['historical_data'].to(DEVICE)
            future_data = batch['future_data'].to(DEVICE)
            edge_index = batch['edge_index'].to(DEVICE)
            edge_attr = batch['edge_attr'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            target_nodes = batch['target_nodes'].to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(future_data, hist_data, edge_index, edge_attr, target_nodes)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                hist_data = batch['historical_data'].to(DEVICE)
                future_data = batch['future_data'].to(DEVICE)
                edge_index = batch['edge_index'].to(DEVICE)
                edge_attr = batch['edge_attr'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                target_nodes = batch['target_nodes'].to(DEVICE)

                outputs = model(future_data, hist_data, edge_index, edge_attr, target_nodes)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_accuracy": val_accuracy, "learning_rate": wandb.config.learning_rate})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improvement = 0

            print(f"New best validation loss: {best_val_loss:.4f}. Saving model... 💾")

            # DataParallel 래핑 여부 확인 후 저장 (문제 2번 해결)
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            
            # Wandb에 최적값 기록
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch + 1
            
        else:
            epochs_no_improvement += 1
            print(f"Validation loss did not improve. Patience: {epochs_no_improvement}/{EARLY_STOPPING_PATIENCE}")

        # 조기 종료 확인
        if epochs_no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break  # 학습 루프 종료



    print("--- Model Training Complete ---")

    # --- 5. Save the trained model ---
    print(f"Training finished. Best model (Val Loss: {best_val_loss:.4f}) is saved at {save_path}")

if __name__ == "__main__":
    train()
