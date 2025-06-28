import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import torch_geometric.data as pygdata
from torch_geometric.graphgym import BondEncoder, AtomEncoder
from composed_encoders import concat_node_encoders
from laplace_pos_encoder import LapPENodeEncoder
from torch_geometric.nn import GCNConv

import math
import torch.nn.functional as F
from einops import rearrange, repeat


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
        h = self.sum(dist_masks,batch.x,batch.batch)
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
    
class Mamba_input(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        return dt,A,B,C,x

# Graph Mamba Layer
class GMBLayer(nn.Module):

    def __init__(
        self,
        local_model: bool,
        dim_hidden: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        drop_rate: int = 0,
        act: str = "full-glu"
    ):
        super().__init__()
        #----------- Local Convolution -----------#
        self.local_model = local_model
        if local_model:
            self.local_model = GCNConv(dim_hidden, dim_hidden)
            self.norm_local = nn.LayerNorm(dim_hidden)
        #----------- Node multiset aggregation -----------#
        self.sum = sumNodeFeatures
        self.mlp1 = MLP1(dim_hidden, expand, drop_rate)
        #----------- Linear Recurrent Network adapted with Mamba -----------#
        # Norm
        self.layer_norm = nn.LayerNorm(dim_hidden)
        # Mamba
        self.self_attn = Mamba_input(d_model=dim_hidden, d_state=d_state, d_conv=d_conv, expand=1)

    def forward(self, batch, dist_masks):
        #----------- Local Convolution -----------#
        if self.local_model:
            out_list = []
            x_skip1 = batch.x
            x = self.norm_local(batch.x)
            x = self.local_model(x,batch.edge_index)
            local = x_skip1 + x
            out_list.append(local)
        #----------- Node multiset aggregation -----------#
        x = self.sum(dist_masks,batch.x,batch.batch)
        # x represents the hidden state after aggregation
        x_skip3 = self.mlp1(x)
        #----------- Mamba block from Graph-Mamba paper -----------#
        # Transpose to (batch_size * num_nodes, seqlen, hidden_dim)
        x = x.transpose(0, 1)
        x = torch.flip(x, dims=[1])
        x = self.layer_norm(x)
        dt,A,B,C,x = self.self_attn(x)
        return dt,A,B,C,x

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self,pos_enc):
        super(FeatureEncoder, self).__init__()
        # Encode integer node features via nn.Embeddings
        if pos_enc:
            NodeEncoder = concat_node_encoders([AtomEncoder, LapPENodeEncoder],['LapPE'])
            self.node_encoder = NodeEncoder(96)
            self.edge_encoder = BondEncoder(96)
        else:
            self.node_encoder = AtomEncoder(88)
            self.edge_encoder = BondEncoder(88)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
    
class GPSModel(nn.Module):
    """
    Multi-scale graph x-former.
    """

    def __init__(self,args):
        self.dim_hidden = args.dim_h
        super().__init__()
        self.dataset = args.name
        self.architecture = args.architecture
        #----------- Node feature Encoder -----------#
        if self.dataset == "peptides-func":
            self.encoder = FeatureEncoder(args.pos_enc)
        elif self.dataset == "CIFAR10":
            self.linearEncoder2 = nn.Linear(args.feature_dimension, args.dim_h)
        self.linearEncoder = nn.Linear(args.dim_h, args.dim_h)
        self.gelu = nn.GELU()
        #----------- Modified Graph Mamba Layer -----------#
        self.layer = GMBLayer(args.local_model, args.dim_h, args.dim_v, drop_rate=args.drop_rate)

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
        dt,A,B,C,x = self.layer(inputs, dist_mask)
        return dt,A,B,C,x
            