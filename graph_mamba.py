import torch
import torch.nn as nn
#from mamba import Mamba,ModelArgs
from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch
from GatedGCN import GatedGCNLayer
import torch_geometric.data as pygdata
from torch_geometric.graphgym import BondEncoder, AtomEncoder
from composed_encoders import concat_node_encoders
from laplace_pos_encoder import LapPENodeEncoder


class MLP1(nn.Module):

    def __init__(self, dim_hidden, expand = 1, drop_rate = 0):
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

    def __init__(self, dim_hidden,dim_v, drop_rate = 0, act = "full-glu"):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.drop_rate = drop_rate
        self.act = act
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_rate)
        if act == "full-glu":
            self.linear1 = nn.Linear(dim_v, dim_hidden)
            self.linear2 = nn.Linear(dim_v, dim_hidden)
        elif act == "half-glu":
            self.linear = nn.Linear(dim_v, dim_hidden)

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


class LSTMLayer(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        dim_v: int,
        expand: int = 1,
        drop_rate: int = 0,
        act: str = "full-glu"
    ):
        super().__init__()
        #----------- Node multiset aggregation -----------#
        self.sum = sumNodeFeatures
        self.mlp1 = MLP1(dim_hidden, expand, drop_rate)
        #----------- Linear Recurrent Network adapted with Mamba -----------#
        # Norm
        self.layer_norm = nn.LayerNorm(dim_hidden)
        # LSTM
        self.self_attn = nn.LSTM(
            input_size=dim_hidden,
            hidden_size=dim_v,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        # MLP
        self.mlp2 = MLP2(dim_hidden, dim_v, drop_rate, act)

    def forward(self, batch, dist_masks):
        #----------- Node multiset aggregation -----------#
        h = self.sum(dist_masks,batch.x,batch.graph_labels)
        # x represents the hidden state after aggregation
        x_skip = self.mlp1(h)
        #----------- Mamba block from Graph-Mamba paper -----------#
        # Transpose to (batch_size * num_nodes, seqlen, hidden_dim)
        x = h.transpose(0, 1)
        x = torch.flip(x, dims=[1])
        x = self.layer_norm(x)
        x,_ = self.self_attn(x)
        x = self.mlp2(x)
        # Reshape back to original dimensions
        x = x.transpose(0, 1)  # Back to (seqlen, batch_size * num_nodes, hidden_dim)
        batch.x = x[-1] + x_skip[0]
        return batch

# Graph Mamba Layer
class GMBLayer(nn.Module):

    def __init__(
        self,
        dim_hidden: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        drop_rate: int = 0,
        act: str = "full-glu"
    ):
        super().__init__()
        #----------- Local Convolution -----------#
        self.local_model = GatedGCNLayer(dim_hidden, dim_hidden,
                                             dropout=drop_rate,
                                             residual=True,
                                             equivstable_pe=True)
        self.norm_local = nn.LayerNorm(dim_hidden)
        #----------- Node multiset aggregation -----------#
        self.sum = sumNodeFeatures
        self.mlp1 = MLP1(dim_hidden, expand, drop_rate)
        #----------- Linear Recurrent Network adapted with Mamba -----------#
        # Norm
        self.layer_norm = nn.LayerNorm(dim_hidden)
        # Mamba
        #model_args = ModelArgs(d_model=dim_hidden,n_layer=4,d_state=d_state, d_conv=d_conv,expand=1)
        #self.self_attn = Mamba(model_args)
        self.self_attn = Mamba(d_model=dim_hidden, d_state=d_state, d_conv=d_conv, expand=1)
        # MLP
        self.mlp2 = MLP2(dim_hidden,dim_hidden, drop_rate, act)
        #----------- Aggregate Local and Global Model -----------#
        self.norm_out = nn.LayerNorm(dim_hidden)

    def forward(self, batch, dist_masks):
        out_list = []
        #----------- Local Convolution -----------#
        x_skip1 = batch.x
        local_out = self.local_model(pygdata.Batch( batch=batch,
                                                    x=batch.x,
                                                    edge_index=batch.edge_index,
                                                    edge_attr=batch.edge_attr,
                                                    pe_EquivStableLapPE=True))
        batch.edge_attr = local_out.edge_attr
        local = self.norm_local(x_skip1 + local_out.x)
        out_list.append(local)
        #----------- Node multiset aggregation -----------#
        x = self.sum(dist_masks,batch.x,batch.graph_labels)
        # x represents the hidden state after aggregation
        x_skip2 = self.mlp1(x)
        #----------- Mamba block from Graph-Mamba paper -----------#
        # Transpose to (batch_size * num_nodes, seqlen, hidden_dim)
        x = x.transpose(0, 1)
        x = torch.flip(x, dims=[1])
        x = self.layer_norm(x)
        x = self.self_attn(x)
        x = x.transpose(0, 1)
        x = x[-1]
        x = self.mlp2(x)
        x = x + x_skip2[0]
        #----------- Aggregate Local and Global Model -----------#
        out_list.append(x)
        batch.x = self.norm_out(sum(out_list))
        return batch

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

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        # Encode integer node features via nn.Embeddings
        NodeEncoder = concat_node_encoders([AtomEncoder, LapPENodeEncoder],['LapPE'])
        self.node_encoder = NodeEncoder(96)
        # Update dim_in to reflect the new dimension fo the node features
        self.dim_in = 96

        # Hard-set edge dim for PNA.
        self.edge_encoder = BondEncoder(96)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
    
class GPSModel(nn.Module):
    """
    Multi-scale graph x-former.
    """

    def __init__(self,architecture,dataset, node_feature_dim, dim_hidden, dim_v, dim_out, num_layers, drop_rate=0):
        self.dim_hidden = dim_hidden
        super().__init__()
        self.dataset = dataset
        self.architecture = architecture
        #----------- Node feature Encoder -----------#
        if dataset == "peptides-func":
            self.encoder = FeatureEncoder(100)
        elif dataset == "CIFAR10":
            self.linearEncoder2 = nn.Linear(node_feature_dim, dim_hidden)
        self.linearEncoder = nn.Linear(dim_hidden, dim_hidden)
        self.gelu = nn.GELU()
        #----------- Modified Graph Mamba Layer -----------#
        layers = []
        for i in range(num_layers):
            if architecture == "GRED-MAMBA":
                layers.append(GMBLayer(dim_hidden, dim_v, drop_rate=drop_rate))
            elif architecture == "LSTM":
                layers.append(LSTMLayer(dim_hidden, dim_v, drop_rate=drop_rate))
        self.layers = nn.Sequential(*layers)
        #----------- Graph Predicition Head -----------#
        self.head = Head(dim_hidden, dim_out)

    def forward(self, inputs, dist_mask, device):
        #----------- Node feature Encoder -----------#
        # Initialize x as zeros
        # Shape: [batch_size, num_nodes, dim_hidden]
        if self.dataset == "peptides-func":
            inputs = self.encoder(inputs)
        elif self.dataset == "CIFAR10":
            inputs.x = self.linearEncoder2(inputs.x)
        
        inputs.x = self.linearEncoder(self.gelu(inputs.x))
        
        #----------- Modified Graph Mamba Layer -----------#
        for layer in self.layers:
            inputs = layer(inputs, dist_mask)
        
        #----------- Graph Predicition Head -----------#
        inputs.x = self.head(inputs.x, inputs.batch)
        return inputs.x
            