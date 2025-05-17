import torch
import torch.nn as nn
#from mamba import Mamba,ModelArgs
from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch

class MLP1(nn.Module):

    def __init__(self, dim_hidden, expand = 1, drop_rate = 0.5):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.expand = expand
        self.drop_rate = drop_rate
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.linear1 = nn.Linear(dim_hidden, expand * dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs):
        x = self.layer_norm(inputs)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + inputs
    
class MLP2(nn.Module):

    def __init__(self, dim_hidden, drop_rate = 0.5, act = "full-glu"):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.drop_rate = drop_rate
        self.act = act
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        if act == "full-glu":
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        elif act == "half-glu":
            self.linear = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = self.dropout(x)
        if self.act == "full-glu":
            x = self.linear1(x) * torch.sigmoid(self.linear2(x))
        elif self.act == "half-glu":
            x = x * torch.sigmoid(self.linear(x))
        x = self.dropout(x)
        return x

def sumNodeFeatures(distance_masks,node_features):
    distance_masks = distance_masks.float()
    return torch.transpose(distance_masks, 0, 1) @ node_features


# Graph Mamba Layer
class GMBLayer(nn.Module):

    def __init__(
        self,
        dim_hidden: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        drop_rate: int = 0.5,
        act: str = "full-glu"
    ):
        super().__init__()
        #----------- Node multiset aggregation -----------#
        self.sum = sumNodeFeatures
        self.mlp1 = MLP1(dim_hidden, expand, drop_rate)
        #----------- Linear Recurrent Network adapted with Mamba -----------#
        # Norm
        self.layer_norm = nn.LayerNorm(dim_hidden)
        # Mamba
        #model_args = ModelArgs(d_model=dim_hidden,n_layer=4,d_state=d_state, d_conv=d_conv,expand=1)
        #self.self_attn = Mamba(model_args)
        self.self_attn = Mamba(d_model=dim_hidden, d_state=16, d_conv=4, expand=1)
        # MLP
        self.mlp2 = MLP2(dim_hidden, drop_rate, act)

    def forward(self, inputs, dist_masks, node_masks):
        #----------- Node multiset aggregation -----------#
        h = self.sum(dist_masks,inputs)
        # x represents the hidden state after aggregation
        x_skip = self.mlp1(h)
        #----------- Mamba block from Graph-Mamba paper -----------#
        # Expand node_masks to match inputs shape for masking
        # Compute graph_label: a 1D vector where each entry indicates the graph index (batch) for each real node
        # Count number of real nodes per graph in the batch
        real_nodes_per_graph = node_masks.sum(dim=1).to(torch.long)  # (batch_size,)
        # For each graph, repeat its index for the number of real nodes it has
        graph_label = torch.cat([
            torch.full((n.item(),), i, dtype=torch.long, device=node_masks.device)
            for i, n in enumerate(real_nodes_per_graph)
        ], dim=0) # (total_real_nodes,)
        # Combine batch and graph dimension of node_masks
        node_masks_flat = node_masks.reshape(-1)  # (batch_size * num_nodes,)
        node_masks_expanded = node_masks.unsqueeze(-1)  # (1, batch_size, num_nodes, 1) # Mask out padded nodes
        #seqlen, batch_size, num_nodes, hidden_dim = x_skip.shape
        # remove padded nodes
        seqlen, batch_size, num_nodes, hidden_dim = x_skip.shape
        x = x_skip.reshape(seqlen, batch_size * num_nodes, hidden_dim)
        x = x[:, node_masks_flat.bool(), :]
        # Reshape to (batch_size, seqlen * num_nodes, hidden_dim)
        #x = x_skip.reshape(seqlen, batch_size * num_nodes, hidden_dim)
        # Transpose to (batch_size * num_nodes, seqlen, hidden_dim)
        x = x.transpose(0, 1)
        x = self.layer_norm(x)
        x = self.self_attn(x)
        x = self.mlp2(x)
        # Reshape back to original dimensions
        x = x.transpose(0, 1)  # Back to (seqlen, batch_size * num_nodes, hidden_dim)
        x = x[0]
        x, _ = to_dense_batch(x, graph_label)
        # Pad x to size 444 in graph dim=1
        if x.shape[1] < 444:
            pad_size = 444 - x.shape[1]
            pad = (0, 0, 0, pad_size)  # (last two dims: (left, right) for each dimension)
            x = torch.nn.functional.pad(x, pad)
        #x = x.reshape(seqlen, batch_size, num_nodes, hidden_dim)
        return x + x_skip[0]

class Head(nn.Module):

    def __init__(self, dim_hidden, dim_output):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_output)

    def forward(self, input, node_masks):
        x = torch.where(node_masks.unsqueeze(-1), input, 0.)
        # aggregating node features -> only "one" graph node left
        x = torch.sum(x, dim=1)
        # task head for prediction
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x
    
full_atom_feature_dims = [119, 5, 12, 12, 10, 6, 6, 2, 2]
full_bond_feature_dims = [5, 6, 2]
    
class GPSModel(nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, node_feature_dim, dim_hidden, dim_out):
        self.dim_hidden = dim_hidden
        super().__init__()
        #----------- Node feature Encoder -----------#
        embedding_modules = []
        for i in range(node_feature_dim):
            embedding = nn.Embedding(full_atom_feature_dims[i], dim_hidden)
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            embedding_modules.append(embedding)
        self.embedding_modules = nn.Sequential(*embedding_modules)
        self.linearEncoder = nn.Linear(dim_hidden, dim_hidden)
        self.gelu = nn.GELU()
        #----------- Modified Graph Mamba Layer -----------#
        layers = []
        for i in range(10):
            layers.append(GMBLayer(dim_hidden))
        self.layers = nn.Sequential(*layers)
        #----------- Graph Predicition Head -----------#
        self.head = Head(dim_hidden, dim_out)

    def forward(self, inputs, node_mask, dist_mask):
        #----------- Node feature Encoder -----------#
        # Initialize x as zeros
        # Shape: [batch_size, num_nodes, dim_hidden]
        x = torch.zeros(inputs.shape[0], inputs.shape[1], self.dim_hidden, device=inputs.device)
        
        # Iterate through each feature dimension
        for i in range(inputs.shape[-1]):
            # Get the current feature
            current_feature = inputs[..., i]
            # Convert to long type for embedding
            current_feature = current_feature.long()
            # Get embedding for this feature
            embedding = self.embedding_modules[i](current_feature)
            # Add to running sum
            x = x + embedding

        x = self.linearEncoder(self.gelu(x))
        
        #----------- Modified Graph Mamba Layer -----------#
        for layer in self.layers:
            x = layer(x, dist_mask, node_mask)
        
        #----------- Graph Predicition Head -----------#
        x = self.head(x, node_mask)
        return x
            