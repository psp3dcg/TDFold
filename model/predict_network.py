'''
Protein Structure Prediction Network
'''
import os
import torch
import torch.nn as nn

from model.build_graph import Build_Graph_Network
from model.parse_protein_info import msa_parser
from model.refine_net import Refine_Network
class Predict_Network(nn.Module):
    def __init__(self, args):
        '''
        Input:
            - args(object):model argument
        '''
        super(Predict_Network, self).__init__()

        self.parse_msa_model = msa_parser(args)
        self.build_graph_model = Build_Graph_Network(args)
        self.refine_model = Refine_Network(args)

    def forward(self, data, test_flag):
        '''main framework

        Input:
            - data(object):input graph data
        Output:
            - xyz(tensor):all atom 3d position
            - model_lddt(tensor):residue sequence LDDT
            - logits(tensor):residue pair distance and torsion angles
        '''
        # extract msa feature
        write_path = os.path.join('msa_pair_feat', data.protein_name[0]+'.pt')
        if not hasattr(data, 'new_t2d'):
            data.new_t2d = []

        if not hasattr(data, 'msa'):
            seq = torch.tensor(data.seq,device='cuda').unsqueeze(0)
        else:
            if len(list(data.msa.size())) < 3:
                data.msa = data.msa.unsqueeze(0)
            seq = data.msa[:,:1]
        # import pdb
        # pdb.set_trace()
        # msa, prob_s, logits, seq1hot, idx = self.parse_msa_model(data.msa, data.xyz_t, data.t1d, data.t0d, new_t2d=data.new_t2d, test_flag=test_flag, name=write_path)
        msa, prob_s, logits, seq1hot, idx = self.parse_msa_model(seq, new_t2d=data.new_t2d, test_flag=test_flag, name=write_path)

        # build protein graph and get backbone position
        bb_xyz, bb_state, atom_coor, atom_feat, node, edge = self.build_graph_model(data, msa, prob_s, seq1hot, idx)

        # refine and get all atom position
        xyz, model_lddt = self.refine_model(bb_xyz, bb_state, atom_coor, atom_feat, \
                                            node, edge, seq1hot, idx, data.CA_atom_index)
        del msa
        del prob_s
        del seq1hot
        del idx

        del bb_xyz
        del bb_state
        del atom_coor
        del atom_feat
        del node
        del edge

        torch.cuda.empty_cache()
        return xyz, model_lddt, logits




        
