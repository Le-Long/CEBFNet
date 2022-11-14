from collections.abc import Sequence

import torch
from torch import nn
from torch import autograd

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from torch_scatter import scatter_add

from torchdrug import core, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from . import layer


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=True,
                 num_beam=10, path_topk=10, path=None):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk

        self.config = AutoConfig.from_pretrained("roberta-base", num_labels=2)

        if path is not None:
            state_dict = torch.load(path + '/pytorch_model.bin', map_location=torch.device('cpu'))
            self.model = AutoModelForSequenceClassification.from_pretrained(path, state_dict=state_dict, config=self.config)
            embedding_dim = double_relation
        else:
            self.model = AutoModel.from_pretrained("roberta-base", config=self.config)
            embedding_dim = 768

        self.model.to(self.device)
        self.model.zero_grad()
        self.model.train()
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])
        self.dropout_m = nn.Dropout(p=0.1)
        self.dropout_q = nn.Dropout(p=0.1)

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                r_index_ext = torch.cat([r_index, r_index], dim=-1)
                pattern = torch.stack([h_index_ext, t_index_ext, r_index_ext], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    @utils.cached
    def bellmanford(self, graph, h_index, t_index, features, separate_grad=False):
        batch_size = features.shape[:2]
        features = torch.flatten(features, end_dim=1).permute((1, 0, 2)).to(torch.long)
        input_ids, input_mask, segment_ids = features
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        logits = outputs[0][:,0,:]
        # query = self.dropout_q(self.query(logits))
        query = torch.reshape(logits, (*batch_size, self.dims[0]))
        index = t_index.unsqueeze(-1).expand_as(query).permute((1, 0, 2))
        boundary = torch.zeros((graph.num_node, batch_size[0], self.dims[0]), device=self.device)
        boundary.scatter_add_(0, index, query.permute((1, 0, 2)))
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                step_graph = graph.clone().requires_grad_()
            else:
                step_graph = graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden

        node_query = boundary
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    def forward(self, graph, h_index, t_index, r_index=None, features=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index.expand_as(t_index), t_index, r_index)

        shape = h_index.shape

        assert (h_index[:, [0]] == h_index).all()
        assert (features[:,[0],[0],[0]].shape == h_index[:, [0]].shape)
        output = self.bellmanford(graph, h_index[:, 0], t_index, features)
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)


        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, h_index[:, 0], features)
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.dropout_m(self.mlp(feature)).squeeze(-1)
        return score
