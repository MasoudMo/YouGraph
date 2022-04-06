import itertools
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, PReLU, ReLU, ELU, Sigmoid
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse, coalesce, sort_edge_index
from conv import AttenConv, GinConv, ExpC, CombC, ExpC_star, CombC_star
import numpy as np


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_vocab,
                 max_seq_len,
                 node_encoder,
                 dropgnn=False,
                 dropgnn_dropout_p=0.0,
                 dropgnn_num_runs=1,
                 nodeskip=False,
                 nodeskip_dropout_p=0.0):

        super(Net, self).__init__()

        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len

        self.node_encoder = node_encoder

        self.convs = torch.nn.ModuleList()
        self.config = config
        if config.methods == 'AC':
            for i in range(config.layers):
                self.convs.append(AttenConv(config.hidden, config.variants))
        elif config.methods == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        elif config.methods[:2] == 'EB':
            for i in range(config.layers):
                self.convs.append(ExpC(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods[:2] == 'EA':
            for i in range(config.layers):
                self.convs.append(ExpC_star(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods == 'CB':
            for i in range(config.layers):
                self.convs.append(CombC(config.hidden, config.variants))
        elif config.methods == 'CA':
            for i in range(config.layers):
                self.convs.append(CombC_star(config.hidden, config.variants))
        else:
            raise ValueError('Undefined gnn layer called {}'.format(config.methods))

        self.JK = JumpingKnowledge(config.JK)

        self.graph_pred_linear_list = torch.nn.ModuleList()
        if config.JK == 'cat':
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(config.hidden * (config.layers + 1), self.num_vocab))
        else:
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(config.hidden, self.num_vocab))

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool

        self.dropout = config.dropout

        # Attributes related to node dropout
        self.dropgnn = dropgnn
        self.dropgnn_dropout_p = dropgnn_dropout_p
        self.dropgnn_num_runs = dropgnn_num_runs
        self.nodeskip = nodeskip
        self.nodeskip_dropout_p = nodeskip_dropout_p
        
        # virtualnode
        if self.config.virtual_node == 'true':
            self.virtualnode_embedding = torch.nn.Embedding(1, config.hidden)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(config.layers - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(config.hidden, 2*config.hidden)
                    ,torch.nn.BatchNorm1d(2*config.hidden)
                    ,torch.nn.PReLU()
                    ,torch.nn.Linear(2*config.hidden, config.hidden)
                    ,torch.nn.BatchNorm1d(config.hidden)
                    ,torch.nn.PReLU()
                    ))

    def forward(self, batched_data):
        '''
            Return:
                A list of predictions.
                i-th element represents prediction at i-th position of the sequence.
        '''
        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth, batched_data.batch

        x = self.node_encoder(x, node_depth.view(-1,))

        if self.nodeskip:
            # Drop nodes randomly
            drop = torch.bernoulli(torch.ones([x.size(0)], device=x.device) * self.nodeskip_dropout_p).bool()
            x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)

            # Augment edges resulting for removal of each node
            dropped_nodes = np.where(drop.detach().cpu().numpy() == 1)[0]
            temp_edge_index = edge_index.detach().cpu().numpy()
            for node in dropped_nodes:

                # Find the nodes where the new edges would go to
                destination_idx = np.where(temp_edge_index[0] == node)[0]
                destination_nodes = temp_edge_index[1, destination_idx]

                # Find the nodes where the new edges originate from
                source_idx = np.where(temp_edge_index[1] == node)[0]
                source_nodes = temp_edge_index[0, source_idx]

                # Add the new edges to edge_index
                new_edges = np.transpose(np.array(list(itertools.product(source_nodes.tolist(),
                                                                         destination_nodes.tolist())), dtype=np.float))
                edge_index = torch.concat((edge_index, torch.tensor(new_edges,
                                                                    dtype=edge_index.dtype,
                                                                    device=edge_index.device)), dim=1)

                # Add the edge attribute for the newly added edges
                edge_attr = torch.concat((edge_attr,
                                          torch.tensor([[2, 2]],
                                                       dtype=edge_attr.dtype,
                                                       device=edge_attr.device).repeat(new_edges.shape[1], 1)), dim=0)

        edge_index, edge_attr = sort_edge_index(edge_index=edge_index,
                                                edge_attr=edge_attr,
                                                num_nodes=batched_data.num_nodes)
        a = edge_index.detach().cpu().numpy()
        b = edge_attr.detach().cpu().numpy()
        # Change data, edge_index, edge_attr and batch to add dropout and account for number of required runs
        if self.dropgnn:

            # Clone the data num_runes times so that runs can be done in parallel
            x = x.unsqueeze(0).expand(self.dropgnn_num_runs, -1, -1).clone()

            # Drop nodes randomly
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.dropgnn_dropout_p).bool()
            x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
            x = x.view(-1, x.size(-1))
            del drop

            # Make proper edge_index and edge_attr for the replicated data
            edge_index = edge_index.repeat(1, self.dropgnn_num_runs) + \
                         torch.arange(self.dropgnn_num_runs,
                                      device=edge_index.device).repeat_interleave(edge_index.size(1)) * \
                         (edge_index.max() + 1)
            edge_attr = edge_attr.repeat((self.dropgnn_num_runs, 1))

            # Replicate batch indices to account for number of runs
            batch = batch.repeat(self.dropgnn_num_runs)

        xs = [x]
        if self.config.virtual_node == 'true':
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        for i, conv in enumerate(self.convs):
            x = xs[i] + virtualnode_embedding[batch] if self.config.virtual_node == 'true' else xs[i]
            x = conv(x, edge_index, edge_attr)
            xs += [x]
            if self.config.virtual_node == 'true' and i < self.config.layers-1:
                virtualnode_embedding_tmp = global_add_pool(xs[i], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[i](virtualnode_embedding_tmp), p=self.dropout, training=self.training)

        nr = self.JK(xs)

        # Need to average over the parallel runs if DropGNN was used
        if self.dropgnn:
            nr = nr.view(self.dropgnn_num_runs, -1, nr.size(-1))
            nr = nr.mean(dim=0)

        nr = F.dropout(nr, p=self.dropout, training=self.training)
        h_graph = self.pool(nr, batched_data.batch)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph) / self.config.T)

        return pred_list

    def __repr__(self):
        return self.__class__.__name__
