
import os
import time
import torch
import torch.nn as nn

from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from utils.aa_info_util import init_res_name_map
from utils.aa_info_util import init_reverse_res_name_map
from utils.aa_info_util import init_atom_num_map
from utils.aa_info_util import init_atom_name_map
from all_atom_fold.loss_func import bond_potential_energy_loss
from all_atom_fold.loss_func import angle_potential_energy_loss
from all_atom_fold.loss_func import dihedral_potential_energy_loss
from torch.optim.lr_scheduler import StepLR

#amino acid short name map
short_name_map = init_res_name_map()

#amino acid short name map in reverse
full_name_map = init_reverse_res_name_map()

#amino acid atom number
atom_num_map = init_atom_num_map()

#amino acid atom name
atom_name_map = init_atom_name_map()


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        '''
        Input:
            - d_model(int):feature dimension
            - eps(float):minimum number avoid divided by zero
        '''
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        '''
        Input:
            - x(tensor):input feature
        Output:
            - x(tensor):output feature after layer normalization
        '''
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2*(x-mean)
        x /= std
        x += self.b_2
        return x

class Local_Energy_Minimizer(nn.Module):
    def __init__(self, init_atom_coord) -> None:
        '''
        Update atom coordinates through potential energy
        Input:
            - init_atom_coord(tensor):initial all atom coordinates
        '''
        super(Local_Energy_Minimizer, self).__init__()

        self.init_coord = nn.Parameter(init_atom_coord)

    def forward(self):
        return self.init_coord


# Define global physical constants
Avogadro = 6.02214086e23
Boltzmann = 1.38064852e-23
class Langevin_Dynamics:
    def __init__(self, temperature, time_step, relax_constant) -> None:
        '''
        Use Langevin dynamics to simulate the motion of particle

        Input:
            - temperature (float): temperature (in Kelvin)
            - time_step (float): simulation timestep (s)
            - relax_constant (float): relaxation constant (in seconds)

        '''

        self.temp  = temperature
        self.relax = relax_constant
        self.time_step  = time_step


    


    def integrate(self, pos, vels, forces, mass,  dt):
        
        ''' 
        A simple forward Euler integrator that moves the system in time 

        Input:
            - pos(numpy array): atomic positions 
            - vels: atomic velocity (ndarray, updated)
        '''
        pos += vels * dt
        # FΔt=mΔv
        vels += forces * dt / mass.unsqueeze(0).T
        
    def computeForce(self, mass, vels, temp, relax, dt):
        
        '''
        Computes the Langevin force for all particles
        
        Input:
            - mass(numpy.ndarray): particle mass 
            - vels(numpy.ndarray): particle velocities 
            - temp(float): temperature 
            - relax(float): thermostat constant 
            - dt(float): simulation timestep 
        Output:
            - forces(numpy.ndarray): forces of particles
        '''

        # FΔt=mΔv
        natoms, ndims = vels.shape

        sigma = torch.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
        noise = torch.randn(natoms, ndims).to(mass.device) * sigma.unsqueeze(0).T

        force = - (vels * mass.unsqueeze(0).T) / relax + noise
        
        return force

    def step(self, n_steps, pos, mass):

        '''
        Input:
            - pos (float): particle position
            - mass (float): particle mass (in Kg)
        '''
        
        vels = torch.randn_like(pos)
        mass /= Avogadro
        pos *= 1e-10

        for step in range(n_steps):
            # Compute all forces
            forces = self.computeForce(mass, vels, self.temp, self.relax, self.time_step)

            # Move the system in time
            self.integrate(pos, vels, forces, mass, self.time_step)

        pos *= 1e+10




class Energy_Minimizer():
    def __init__(self,             
            first_iter_steps=1, 
            second_iter_steps=1, 
            learning_rate=0.001,
            weight_decay=0.0005,
            device='cuda') -> None:

        '''
        Refine the all atom structure by energy minimization

        Input:
            - first_iter_steps(int): the iteration steps for main refinement
            - second_iter_steps(int): the iteration steps for refinement after langevin dynamics simulation
            - learning_rate(float): learning rate of optimizer
            - weight_decay(float): weight of L2 
            - device(str): the working device, cuda or cpu
        '''

        self.first_iter_steps = first_iter_steps
        self.second_iter_steps = second_iter_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_grad = 0.1
        self.atom_mass = torch.tensor([12.0107,14.0067,15.9994,32.0655], device=device)
        self.device = device

    def set_coord_grad_mask(self, init_coord, N_atom_index):

        '''
        Make the coordinate gradient mask, the gradient of backbone is set to 0

        Input:
            - init_coord(tensor): initial all atom coordinates
            - N_atom_index(tensor): the index of Nitrogen atoms
        Output:
            - atom_mask(tensor): the 0-1 mask tensor of atoms
        '''

        prev_index = 0
        atom_mask = []
        for i, index in enumerate(N_atom_index):
            if i == 0:
                continue
            if i == N_atom_index.shape[0] - 1:
                current_index = init_coord.shape[0]
            else:
                current_index = index
            res_coord = init_coord[prev_index:current_index]
            bb_coord = res_coord[:3, :]
            sc_coord = res_coord[3:, :]
            
            bb_mask = torch.zeros(*bb_coord.shape)
            sc_mask = torch.ones(*sc_coord.shape)

            atom_mask.append(bb_mask)
            atom_mask.append(sc_mask)

            prev_index = index

        atom_mask = torch.cat(atom_mask)
        return atom_mask

    def findNearestDistance(self, init_coord):
        
        '''
        Given a set of newly added atoms, find the closest distance between 
            one of those atoms and another atom.

        Input:
            - init_coord(tensor): initial all atom coordinates
        Output:
            - nearest(tensor): the closest distance between all atom pairs
        '''


        device = init_coord.device
        prev_index = 0
        nearest = 10000.

        all_atom_num = init_coord.shape[0]

        all_coord =  init_coord.unsqueeze(1).repeat(1, all_atom_num, 1)
        all_coord_T = all_coord.permute(1,0,2)
        all_direct_vec = all_coord - all_coord_T

        # make diagonal elements huge enough, pervent calculating self-distance 
        huge_padding = torch.eye(all_atom_num)*100.
        huge_padding = huge_padding.to(device)
        dist = torch.norm(all_direct_vec, dim=-1) + huge_padding

        nearest = torch.min(dist)

        return nearest




    def iteration(self, iteration_steps, optimizer, update_coord, data,
                protein_name, min_grad):
        
        '''
        Single step optimization process

        Input:
            - iteration_steps(int): iteration steps for refinement
            - optimizer(torch.optimizer): optimizer in pytorch
            - update_coord(tensor): all atom coordinates
            - data(torch geometric Data object): protein data
            - protein_name(str): the PDB code of protein
            - min_grad(float): the gradient threshold. If the calculated gradient 
                is less than the threshold, the optimization stops
        Output:
            - update_coord(tensor): updated all atom coordinates
        '''

        scheduler = StepLR(optimizer, 1, gamma=0.5)
        min_loss = 1e+5
        patience = 0
        for epoch in range(iteration_steps):
            optimizer.zero_grad()

            bond_energy_loss = bond_potential_energy_loss(update_coord, data.bond_value, data.bond_index.T)
            angle_energy_loss = angle_potential_energy_loss(update_coord, data.angle_value, data.angle_index)
            dihedral_energy_loss = dihedral_potential_energy_loss(update_coord, data.dihedral_value, data.dihedral_index)

            sum_loss = sum([0.01*bond_energy_loss, 0.01*angle_energy_loss,0.001*dihedral_energy_loss])
            sum_loss.backward()
            optimizer.step()
    
            if update_coord.grad.norm() < min_grad:
                break

        
        return update_coord

    def minimize(self, data):
        
        '''
        Single step optimization process

        Input:
            - data(torch geometric Data object): protein data
        Output:
            - pred_coor_list(list): final all atom coordinates
        '''


        data = data.to(self.device)
        update_coord = data.x.clone()
        protein_name = data.protein_name
        N_atom_index = torch.subtract(data.CA_atom_index, 1)
        all_atom_mask = self.set_coord_grad_mask(update_coord, N_atom_index).to(self.device)

        def zero_unlearnable(grad):
            return grad * all_atom_mask

        update_coord.requires_grad = True
        update_coord.register_hook(zero_unlearnable)
        
        optimizer = torch.optim.AdamW([update_coord], lr=self.learning_rate, weight_decay=self.weight_decay)

        update_coord = self.iteration(self.first_iter_steps, optimizer, update_coord, 
                                data, protein_name, self.min_grad)

        

        # Langevin dynamics to prevent very little distance
        # temperture kalvin, time step, relax constant(friction coeff, lambda)
        integrator = Langevin_Dynamics(300, 5*1e-15, 1e-13)
        coord_for_langevin = update_coord.clone().detach()
        coord_temp = coord_for_langevin.clone()
        nearest = self.findNearestDistance(update_coord.detach())
        if nearest < 0.13:
            for i in range(50):
                integrator.step(200, coord_for_langevin, self.atom_mass[data.atom_type_index])
                d = self.findNearestDistance(coord_for_langevin)
                if d > nearest:
                    coord_temp = coord_for_langevin.clone()
                    nearest = d
                    if nearest >= 0.13:
                        break
            coord_for_langevin = coord_temp

        # final energy minimization
        update_coord = coord_for_langevin
        update_coord.requires_grad = True
        update_coord.register_hook(zero_unlearnable)

        optimizer = torch.optim.AdamW([update_coord], lr=0.0001, weight_decay=0.0005)
        update_coord = self.iteration(self.second_iter_steps, optimizer, update_coord, 
                                data, protein_name, self.min_grad)

        
        

        # output final coord
        final_pred_coord = update_coord.detach()
        pred_coor_list = final_pred_coord.tolist()

        return pred_coor_list

 

