
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

from data_processing.dataset import NGSIMDataset
from models.model import DualViewInteractionAwareModel
from train import collate_fn, BATCH_SIZE, DEVICE, DATA_PATH, HISTORY_FRAMES, FUTURE_FRAMES, FEATURE_DIM, LSTM_HIDDEN, NUM_HEADS, NUM_CLASSES

def evaluate():
    print("--- Starting Model Evaluation ---")
    MODEL_PATH = "../checkpoints/lane_change_model.pth"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train.py to train and save the model first.")
        return

    # --- 1. Load Data ---
    # Use the same validation set for consistency if possible.
    # For simplicity, we create a new split here. For rigorous results, save the split indices.
    print(f"Loading data from {DATA_PATH}...")
    try:
        full_dataset = NGSIMDataset(data_dir=DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Use a fixed seed to get the same validation set as in training
    torch.manual_seed(42)
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    torch.manual_seed(torch.initial_seed()) # Reset seed

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0) # num_workers=0 for simplicity in eval
    print(f"Data loaded. Using {len(val_dataset)} samples for validation.")

    # --- 2. Load Model ---
    model = DualViewInteractionAwareModel(
        feature_dim=FEATURE_DIM,
        lstm_hidden=LSTM_HIDDEN,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Inference Loop ---
    all_labels = []
    all_preds = []
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            hist_data = batch['historical_data'].to(DEVICE)
            future_data = batch['future_data'].to(DEVICE)
            edge_index = batch['edge_index'].to(DEVICE)
            edge_attr = batch['edge_attr'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            target_nodes = batch['target_nodes'].to(DEVICE)

            outputs = model(future_data, hist_data, edge_index, edge_attr, target_nodes)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # --- 4. Calculate and Print Metrics ---
    print("\n--- Evaluation Results ---")
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    # Use 'macro' average to treat all classes equally, good for imbalance
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score (Macro): {f1:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")

    # --- 5. Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    class_names = ['Keep', 'Left', 'Right'] # Assuming 0: Keep, 1: Left, 2: Right
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Optional: Plot confusion matrix
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.savefig("../evaluation_results.png")
        print(f"\nConfusion matrix plot saved to evaluation_results.png")
    except Exception as e:
        print(f"\nCould not plot confusion matrix. Error: {e}")
        print("Please ensure seaborn and matplotlib are installed (`pip install seaborn matplotlib`)")

if __name__ == "__main__":
    evaluate()
