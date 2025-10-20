import torch
import torch.nn as nn
from .modules import (
    FutureInteractionModule, 
    HistoricalInteractionModule, 
    InteractionFusionModule
)

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

    def forward(self, future_data, historical_data, edge_index, edge_attr, target_nodes):
        """
        Defines the forward pass of the model.
        
        NOTE: This is a simplified forward pass. The actual implementation will need to 
        handle complex batching of graph data, likely using `torch_geometric.data.Batch`.
        The input tensors are assumed to be properly formatted node features for a batch of graphs.
        
        Args:
            future_data (Tensor): Batch of features for the future interaction module.
                                  Shape: (num_nodes_in_batch, sequence_length, feature_dim)
            historical_data (Tensor): Batch of features for the historical interaction module.
                                      Shape: (num_nodes_in_batch, sequence_length, feature_dim)
            edge_index (Tensor): The graph connectivity for the GraphTransformer.
                                 Shape: (2, num_edges_in_batch)
            edge_attr (Tensor): Edge attributes for the edge-biased attention.
                                Shape: (num_edges_in_batch, edge_feature_dim)
            target_nodes (Tensor): Indices of the target nodes in the batch.
                                   Shape: (batch_size,)
        """
        
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
