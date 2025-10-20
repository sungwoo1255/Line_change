import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, MessagePassing
from torch_geometric.utils import softmax


class GraphTransformer(nn.Module):
    """
    Graph Transformer layer using TransformerConv from PyTorch Geometric.
    """
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        # Use concat=False to ensure the output dimension is `out_channels`
        # and not `out_channels * num_heads`.
        self.conv = TransformerConv(in_channels, out_channels, heads=num_heads, concat=False, dropout=0.1)

    def forward(self, x, edge_index):
        # Perform the actual graph attention operation
        return self.conv(x, edge_index)

class FutureInteractionModule(nn.Module):
    """
    Implements the Future Interaction Module.
    As described in the paper: Input -> LSTM -> Multi-Head Attention -> GraphTransformer.
    It processes features related to 'perceived safety' (TTC, etc.).
    """
    def __init__(self, feature_dim, lstm_hidden, num_heads):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.mha = nn.MultiheadAttention(lstm_hidden, num_heads, batch_first=True, dropout=0.1)
        self.graph_transformer = GraphTransformer(lstm_hidden, lstm_hidden)
        self.norm = nn.LayerNorm(lstm_hidden)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, seq_len, feature_dim]
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step from LSTM as a summary of the sequence
        # lstm_out shape: [num_nodes, seq_len, lstm_hidden]
        last_step_out = lstm_out[:, -1, :] # Shape: [num_nodes, lstm_hidden]

        # MHA and GraphTransformer now operate on 2D tensors
        attn_out, _ = self.mha(last_step_out, last_step_out, last_step_out)
        graph_out = self.graph_transformer(self.norm(attn_out), edge_index)
        return graph_out

class HistoricalInteractionModule(nn.Module):
    """
    Implements the Historical Interaction Module.
    The paper states it's a 'similar branch' to the Future module.
    It processes historical trajectory data.
    """
    def __init__(self, feature_dim, lstm_hidden, num_heads):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.mha = nn.MultiheadAttention(lstm_hidden, num_heads, batch_first=True, dropout=0.1)
        self.graph_transformer = GraphTransformer(lstm_hidden, lstm_hidden)
        self.norm = nn.LayerNorm(lstm_hidden)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, seq_len, feature_dim]
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step from LSTM as a summary of the sequence
        last_step_out = lstm_out[:, -1, :] # Shape: [num_nodes, lstm_hidden]

        # MHA and GraphTransformer now operate on 2D tensors
        attn_out, _ = self.mha(last_step_out, last_step_out, last_step_out)
        graph_out = self.graph_transformer(self.norm(attn_out), edge_index)
        return graph_out

class EdgeBiasedSelfAttention(MessagePassing):
    """
    Custom Graph Attention Layer that incorporates edge attributes (biases)
    into the attention score calculation, as described in the paper.
    """
    def __init__(self, in_channels, out_channels, heads=8, edge_dim=2, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.heads = heads
        self.out_channels = out_channels
        self.scale = (out_channels // heads) ** -0.5

        self.to_qkv = nn.Linear(in_channels, out_channels * 3, bias=False)
        self.to_edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.to_out = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, edge_dim]

        # Project to Q, K, V and reshape for multi-head attention
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(-1, self.heads, self.out_channels // self.heads), qkv)

        # Project edge attributes to get a bias term for each head
        edge_bias = self.to_edge_bias(edge_attr)  # [E, heads]

        # Start message passing
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_bias=edge_bias, size=None)

        # Reshape and final linear projection
        out = out.view(-1, self.out_channels)
        return self.to_out(out)

    def message(self, q_i, k_j, v_j, edge_bias, index, ptr, size_i):
        # q_i, k_j, v_j: [E, heads, head_dim]
        # edge_bias: [E, heads]

        # 1. Calculate raw attention score
        attn_score = (q_i * k_j).sum(dim=-1) * self.scale

        # 2. Add the pre-computed edge bias
        biased_attn_score = attn_score + edge_bias

        # 3. Apply softmax to get attention weights
        attn_weights = softmax(biased_attn_score, index, ptr, size_i)

        # 4. Weight values by attention and return
        return v_j * attn_weights.unsqueeze(-1)


class InteractionFusionModule(nn.Module):
    """
    Implements the Interaction Fusion Module.
    It uses an edge-biased self-attention layer and a final classifier.
    """
    def __init__(self, in_features, num_classes=3):
        super().__init__()
        self.edge_biased_attention = EdgeBiasedSelfAttention(in_features, in_features, heads=8, edge_dim=2)
        self.norm = nn.LayerNorm(in_features)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        # x is the fused_features from the two branches
        attn_out = self.edge_biased_attention(x, edge_index, edge_attr)
        # Add & Norm, a standard Transformer block operation
        x = self.norm(x + attn_out)
        return self.classifier(x)
