'''
build residue graph and atom graph,
compute initial backbone atom postion and
sidechain atom position
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Transformer import LayerNorm
from model.InitStrGenerator import make_graph
from model.InitStrGenerator import get_seqsep, UniMPBlock
from model.Attention_module_w_str import get_bonded_neigh
from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp



# class Atom_Net(torch.nn.Module):
#     # atom(as node) graph network
#     def __init__(self, num_features, num_classes=3, nhid=128, dropout_ratio=0.5):

#         '''
#         Input:
#             - num_features(int):feature dimension
#             - num_classes(int):3d position dimension
#             - nhid(int):hidden layer dimension
#             - dropout_ratio(float):dropout ratio
#         '''
#         super(Atom_Net, self).__init__()
		
#         self.nhid = nhid
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.dropout_ratio = dropout_ratio
		

#         self.conv1 = GraphConv(self.num_features, self.nhid)
#         self.conv2 = GraphConv(self.nhid, self.nhid)
#         self.conv3 = GraphConv(self.nhid, self.nhid)

#         self.lin1 = torch.nn.Linear(self.nhid, self.nhid//2)
#         self.lin2 = torch.nn.Linear(self.nhid//2, self.nhid//4)
#         self.lin3 = torch.nn.Linear(self.nhid//4, self.num_classes)

#         self.relu = nn.PReLU(num_parameters=1, init=0.25)

  
#     def forward(self, data):
#         '''
#         Input:
#             - data(object):input data(node feature, adjacent matrix)
#         Output:
#             - x(tensor):node position
#             - x_ori(tensor):node feature
#         '''
#         x, edge_index = data.x, data.edge_index

#         x1 = self.relu(self.conv1(x, edge_index))

#         x2 = self.relu(self.conv2(x1, edge_index))

#         x3 = self.relu(self.conv3(x2, edge_index))

#         x_ori = x1 + x2 + x3

#         # node feature mlp
#         x = self.relu(self.lin1(x_ori))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = self.relu(self.lin2(x))
#         x = self.lin3(x)

#         torch.cuda.empty_cache()
   
#         return x, x_ori

    
class Res_Network(nn.Module):
    # residue(as node) graph network
    def __init__(self, 
                 node_dim_in=64, 
                 node_dim_hidden=64,
                 edge_dim_in=128, 
                 edge_dim_hidden=64, 
                 state_dim=8,
                 nheads=4, 
                 nblocks=3, 
                 dropout=0.5,
                 num_features=43, 
                 num_classes=3, 
                 nhid=64, 
                 dropout_ratio=0.5):

        '''
        Input:
            - node_dim_in(int):node feature input dimension
            - node_dim_hidden(int):node feature hidden dimension
            - edge_dim_in(int):edge feature input dimension
            - edge_dim_hidden(int):edge feature hidden dimension
            - state_dim(int):state dimenison
            - nheads(int):number of attention heads
            - nblocks(int):number of blocks
            - dropout(float):dropout ratio
        '''
        super(Res_Network, self).__init__()

        # embedding layers for node and edge features
        self.norm_node = LayerNorm(node_dim_in)
        self.norm_edge = LayerNorm(edge_dim_in)

        self.embed_x = nn.Sequential(nn.Linear(node_dim_in+21, node_dim_hidden), LayerNorm(node_dim_hidden))
        self.embed_e = nn.Sequential(nn.Linear(edge_dim_in+2, edge_dim_hidden), LayerNorm(edge_dim_hidden))
        
        # graph transformer
        blocks = [UniMPBlock(node_dim_hidden,edge_dim_hidden,nheads,dropout) for _ in range(nblocks)]
        self.transformer = nn.Sequential(*blocks)

        self.final_res_gnn = UniMPBlock(node_dim_hidden,edge_dim_hidden,nheads,dropout)

        # self.res_graph_net_1 = UniMPBlock(node_dim_hidden,edge_dim_hidden,nheads,dropout)
        # self.res_graph_net_2 = UniMPBlock(node_dim_hidden,edge_dim_hidden,nheads,dropout)
        # self.res_graph_net_3 = UniMPBlock(node_dim_hidden,edge_dim_hidden,nheads,dropout)

        self.nhid = nhid
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
    
        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.conv3 = GraphConv(self.nhid, self.nhid)

        self.relu = nn.PReLU(num_parameters=1, init=0.25)

        # concat feature encoder
        # self.res_atom_encoder = nn.Linear(128, 64)
        # self.atom_encoder = nn.Linear(128, 64)
        # self.atom_encoder_2 = nn.Linear(128, 64)
        # self.res_encoder  = nn.Linear(128, 64)
        
        # outputs
        self.get_xyz = nn.Linear(node_dim_hidden,9)
        self.norm_state = LayerNorm(node_dim_hidden)
        self.get_state = nn.Linear(node_dim_hidden, state_dim)

        self.s_func = torch.nn.Sigmoid()
        self.proj_atom = nn.Linear(self.nhid, node_dim_hidden)
        self.softmax = torch.nn.Softmax(dim=0)
        self.res_atom_encoder = nn.Linear(node_dim_hidden*2, node_dim_hidden)
    
    def make_batch(self, data):
        N_atom_index = torch.subtract(data.CA_atom_index, 1)
        for i in range(N_atom_index.shape[0]):
            if i == 0:
                batch_tensor = torch.ones(N_atom_index[i+1] - N_atom_index[i], dtype=torch.long)*i
            elif i == (N_atom_index.shape[0] - 1):
                new_batch = torch.ones(data.x.shape[0] - N_atom_index[i], dtype=torch.long)*i
                batch_tensor = torch.cat([batch_tensor, new_batch], dim=-1)
            else:
                new_batch = torch.ones(N_atom_index[i+1] - N_atom_index[i], dtype=torch.long)*i
                batch_tensor = torch.cat([batch_tensor, new_batch], dim=-1)

        return batch_tensor.cuda()

    def extend_feat(self, res_feat, batch):
        new_res_feat = list()
        for i in batch:
            new_res_feat.append(res_feat[i,:])

        return torch.cat(new_res_feat)



    def cos_similarity(self, x, y):
        "Returns the cosine distance batchwise"
        # x is the image feature: bs * d * m * m
        # y is the audio feature: bs * d * nF
        # return: bs * n * m
        # torch.view需要操作连续的Tensor，transpose、permute操作新建tensor并在新的信息中重新指定stride,torch.view方法约定不修改数组本身，只是使用新的形状查看数据.
        # torch.norm(input,p,dim,out=None,keepdim=False)->Tensor返回输入张量给定维度dim上的每行p范数，keepdim(bool)保持输出的维度
        # x = x.div(torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
        # y = y.div(torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)
        # cos_dis = torch.bmm(x,torch.transpose(y,1,2))  # .transpose(1,2)
        # import pdb
        # pdb.set_trace()

        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.matmul(x,torch.transpose(y,0,1))  # .transpose(1,2)

        return cos_dis

    def forward(self, seq1hot, idx, node, edge, data):
        '''
        Input:
            - seq1hot(tensor):residue sequence onehot vec
            - idx(tensor):residue index
            - node(tensor):residue graph node feature
            - edge(tensor):residue graph node interaction
        Output:
            - xyz(tensor):backbone atom 3d position
            - state(tensor):sequence state information
        '''
        # import pdb
        # pdb.set_trace()
        B, L = node.shape[:2]
        node = self.norm_node(node)
        edge = self.norm_edge(edge)
        
        node = torch.cat((node, seq1hot), dim=-1)
        node = self.embed_x(node)

        seqsep = get_seqsep(idx) 
        neighbor = get_bonded_neigh(idx)
        edge = torch.cat((edge, seqsep, neighbor), dim=-1)
        edge = self.embed_e(edge)
        

        # batch = self.make_batch(data)
        G = make_graph(node, idx, edge)
        # res feat
        Gout = self.transformer(G)
        # import pdb
        # pdb.set_trace()
        # atom feat
        x, edge_index = data.x, data.edge_index
        x1 = self.relu(self.conv1(x, edge_index))
        x2 = self.relu(self.conv2(x1, edge_index))
        x3 = self.relu(self.conv3(x2, edge_index))        
        x3 = self.proj_atom((x1+x2+x3))
        # x3 = self.proj_atom(x3)
        cos_sim = self.cos_similarity(Gout.x, x3)

        # new_cos_sim = self.s_func(cos_sim)
        new_cos_sim = self.softmax(cos_sim)

        # import pdb
        # pdb.set_trace()

        filter_matrix = torch.bernoulli(new_cos_sim)
        new_cos_sim = filter_matrix * new_cos_sim

        new_x3 = torch.matmul(new_cos_sim, x3)

        Gout.x = self.res_atom_encoder(torch.cat([Gout.x, new_x3], dim=-1))

        Gout = self.final_res_gnn(Gout)

        # new_cos_sim = torch.nn.Softmax(cos_sim, dim=1)
        # new_atom_feat = torch.matmul(new_cos_sim.T, x3)



        # new_res_feat = self.extend_feat(Gout.x, batch)

        # new_feat = torch.cat([new_res_feat, x3])
        # new_feat = self.res_atom_encoder(new_feat)

        # new_atom_feat = torch.cat([x3, new_feat])
        # new_atom_feat = self.atom_encoder(new_atom_feat)

        # new_atom_feat = torch.cat([gmp(new_atom_feat, batch), gap(new_atom_feat, batch)], dim=-1)
        # new_atom_feat = self.atom_encoder_2(new_atom_feat)

        # new_res_feat  = torch.cat([Gout.x, new_atom_feat] )
        # new_res_feat  = self.res_encoder(new_res_feat)
        
        # xyz = self.get_xyz(new_res_feat)
        # state = self.get_state(self.norm_state(new_res_feat))
        
        xyz = self.get_xyz(Gout.x)
        state = self.get_state(self.norm_state(Gout.x))

        torch.cuda.empty_cache()
        return xyz.reshape(B, L, 3, 3) , state.reshape(B, L, -1)


class Build_Graph_Network(nn.Module):
    def __init__(self, args):
        '''
        Input:
            - args(object):model argument
        '''
        super(Build_Graph_Network, self).__init__()
        self.residue_graph_network = Res_Network(node_dim_in=args.d_msa, node_dim_hidden=args.d_hidden,
                                                   edge_dim_in=args.d_hidden*2, edge_dim_hidden=args.d_hidden,
                                                   state_dim=args.l0_out_feat,
                                                   nheads=4, nblocks=3, dropout=args.p_drop)
        # self.atom_graph_network = Atom_Net(num_features=args.num_features, 
        #                               num_classes = args.num_classes,
        #                               nhid = args.nhid,
        #                               dropout_ratio = args.dropout_ratio)

        self.proj_edge = nn.Linear(args.edge_d_pair, args.d_hidden*2)
        

    def forward(self, data, node, edge, seq1hot, idx):
        '''
        Input:
            - data(object):atom graph node feature and adjacent matrix
            - node(tensor):residue graph node feature
            - edge(tensor):residue graph node interaction
            - seq1hot(tensor):residue sequence onehot vec
            - idx(tensor):residue index
        Output:
            - bb_xyz(tensor):backbone atom position
            - bb_state(tensor):backbone atom position
            - atom_coor(tensor):sidechain atom graph node position
            - atom_feat(tensor):sidechain atom graph node feat
            - node(tensor):residue graph node feature
            - edge(tensor):residue graph node interaction
        '''

        edge = self.proj_edge(edge)

        bb_xyz, bb_state = self.residue_graph_network(seq1hot, idx, node, edge, data)

        # atom_coor, atom_feat = self.atom_graph_network(data)
        atom_coor, atom_feat = [], []

        return bb_xyz, bb_state, atom_coor, atom_feat, node, edge