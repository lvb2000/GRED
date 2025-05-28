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

def sumNodeFeatures(distance_masks,node_features,graph_labels):
    try:
        dense_features, mask = to_dense_batch(node_features, graph_labels)
        distance_masks = distance_masks.float()
        aggregated_features = torch.transpose(distance_masks, 0, 1) @ dense_features
        # Apply mask to the second dimension (nodes) of aggregated_features
        aggregated_features = aggregated_features[:, mask, :]
        return aggregated_features
    except Exception as e:
        print("Exception in sumNodeFeatures:", e)
        print("distance_masks shape:", distance_masks.shape)
        print("node_features shape:", node_features.shape)
        print("graph_labels shape:", graph_labels.shape)
        try:
            dense_features, mask = to_dense_batch(node_features, graph_labels)
            print("dense_features shape:", dense_features.shape)
            print("mask shape:", mask.shape)
        except Exception as e2:
            print("Exception during to_dense_batch:", e2)
        try:
            distance_masks_f = distance_masks.float()
            print("distance_masks (after float) shape:", distance_masks_f.shape)
        except Exception as e3:
            print("Exception during distance_masks.float():", e3)
        try:
            aggregated_features = torch.transpose(distance_masks.float(), 0, 1) @ dense_features
            print("aggregated_features (before masking) shape:", aggregated_features.shape)
            aggregated_features = aggregated_features[:, mask, :]
            print("aggregated_features (after masking) shape:", aggregated_features.shape)
        except Exception as e4:
            print("Exception during aggregation:", e4)
        raise


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

    def forward(self, inputs, dist_masks, graph_labels):
        #----------- Node multiset aggregation -----------#
        h = self.sum(dist_masks,inputs,graph_labels)
        # x represents the hidden state after aggregation
        x_skip = self.mlp1(h)
        #----------- Mamba block from Graph-Mamba paper -----------#
        # Transpose to (batch_size * num_nodes, seqlen, hidden_dim)
        x = h.transpose(0, 1)
        x = self.layer_norm(x)
        x = self.self_attn(x)
        x = self.mlp2(x)
        # Reshape back to original dimensions
        x = x.transpose(0, 1)  # Back to (seqlen, batch_size * num_nodes, hidden_dim)
        return x[0] + x_skip[0]

class Head(nn.Module):

    def __init__(self, dim_hidden, dim_output):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_output)

    def forward(self, input, graph_labels):
        x , _ = to_dense_batch(input, graph_labels)
        # aggregating node features -> only "one" graph node left
        x = torch.sum(x, dim=1)
        # task head for prediction
        x = self.gelu(self.linear1(x))
        x = self.linear2(x)
        return x
    
full_atom_feature_dims = [119, 5, 12, 12, 10, 6, 6, 2, 2]
full_bond_feature_dims = [5, 6, 2]
    
class GPSModel(nn.Module):
    """
    Multi-scale graph x-former.
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

    def forward(self, inputs, dist_mask, device):
        #----------- Node feature Encoder -----------#
        # Initialize x as zeros
        # Shape: [batch_size, num_nodes, dim_hidden]
        x = torch.zeros(inputs.x.shape[0], self.dim_hidden, device=device)
        
        # Iterate through each feature dimension
        for i in range(inputs.x.shape[-1]):
            # Get the current feature
            current_feature = inputs.x[..., i]
            # Convert to long type for embedding
            current_feature = current_feature.long()
            # Get embedding for this feature
            embedding = self.embedding_modules[i](current_feature)
            # Add to running sum
            x = x + embedding

        x = self.linearEncoder(self.gelu(x))
        
        #----------- Modified Graph Mamba Layer -----------#
        for layer in self.layers:
            x = layer(x, dist_mask, inputs.batch)
        
        #----------- Graph Predicition Head -----------#
        x = self.head(x, inputs.batch)
        return x
            