import torch
import torch.nn as nn
from .modules import (
    FutureInteractionModule, 
    HistoricalInteractionModule, 
    InteractionFusionModule
)
# [추가] PyG의 Batch 타입을 임포트 (타입 힌트를 위해)
from torch_geometric.data import Batch

class DualViewInteractionAwareModel(nn.Module):
    """
    Main model architecture assembling the different modules as described in the paper:
    'Dual-View Interaction-Aware Lane Change Prediction for Autonomous Driving'
    
    This class defines the overall structure and data flow.
    """
    def __init__(self, feature_dim, lstm_hidden, num_heads=8, num_classes=3):
        super().__init__()
        
        # A simple input projection layer can be useful
        self.input_proj = nn.Linear(feature_dim, lstm_hidden)

        # Module to process future interactions based on perceived safety
        self.future_module = FutureInteractionModule(
            feature_dim=lstm_hidden, 
            lstm_hidden=lstm_hidden, 
            num_heads=num_heads
        )
        
        # Module to process historical interactions from vehicle trajectories
        self.historical_module = HistoricalInteractionModule(
            feature_dim=lstm_hidden, 
            lstm_hidden=lstm_hidden, 
            num_heads=num_heads
        )
        
        # Module to fuse the features from both views and classify the intention
        # The input dimension is doubled because we concatenate future and historical features
        self.fusion_module = InteractionFusionModule(
            in_features=lstm_hidden * 2, 
            num_classes=num_classes
        )

# [수정] forward 함수의 시그니처를 변경
    def forward(self, batch: Batch):
        """
        Defines the forward pass of the model.
        [수정]
        이제 PyG 'Batch' 객체를 직접 인수로 받습니다.
        
        Args:
            batch (torch_geometric.data.Batch): 
                GPU로 이동된 PyG 배치 객체.
                필요한 모든 속성(historical_data, edge_index 등)을 포함합니다.
        """
        
        # [수정] Batch 객체에서 변경된 속성 이름으로 텐서를 추출
        future_data = batch.future_x   # <--- 'batch.future_data'에서 변경
        historical_data = batch.x      # <--- 'batch.historical_data'에서 변경
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        target_nodes = batch.ptr[:-1]
        
        # Project input features to the model's hidden dimension
        projected_future = self.input_proj(future_data)
        projected_historical = self.input_proj(historical_data)

        # Process each view through its respective module
        future_features = self.future_module(projected_future, edge_index)
        historical_features = self.historical_module(projected_historical, edge_index)
        
        # Concatenate the features from both views for fusion
        # This combines the 'dual-view' information for each node in the graph.
        combined_features = torch.cat([future_features, historical_features], dim=-1)
        
        # Fuse the features and get the final class logits
        # The fusion module will operate on each node's combined features.
        logits = self.fusion_module(combined_features, edge_index, edge_attr)
        
        # Get the logits only for the target vehicles in the batch
        target_logits = logits[target_nodes]
        
        return target_logits
