import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from ocpmodels.datasets.embeddings import KHOT_EMBEDDINGS, QMOF_KHOT_EMBEDDINGS
from ocpmodels.models.base import BaseModel

import yaml
from nequip.model import model_from_config

from nequip.data import AtomicData, AtomicDataDict

@registry.register_model("allegro")
class Allegro(BaseModel):
    r"""Our wrapper for the Allegro model.

    Args:
        num_atoms (int): Number of atoms. <--- UNUSED FOR ALLEGRO, 
                                               SPECIFIED IN MODEL CONFIG
        bond_feat_dim (int): Dimension of bond features. <--- UNUSED FOR ALLEGRO,
                                                        SPECIFIED IN MODEL CONFIG
        num_targets (int): Number of targets to predict. <--- UNUSED FOR ALLEGRO,
                                                        SPECIFIED IN MODEL CONFIG

        config_path (str): Path to the config file. 
        Check out allegro/allegro_model_config.yaml for an example.

        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions. 
                                        <--- THIS ONE IS USED, IT IS IMPORTANT --- >
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
                                        <--- THIS ONE IS USED, IT IS IMPORTANT--- >
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly. <--- Whether you use this one depends on whether you preprocess the graph with edges or not. 
                                                                                                Set to false if preprocessed with edges.
            (default: :obj:`False`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.            <--- Default is 6.0 Angstroms. Useful for the calculations in the `if self.use_pbc:` block.
            (default: :obj:`10.0`)
    """

    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        config_path='allegro/allegro_model_config.yaml',
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        cutoff=6.0,

    ):
        super(Allegro, self).__init__(num_atoms, bond_feat_dim, num_targets)
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph


        """
        Load allegro model config from yaml file and create it.
        """

        with open(config_path, "r") as stream:
            try:
                config  = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.model = model_from_config(config)


    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        # Get node features
        pos = data.pos

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors


        # I am unsure if this is the right way to do this.
        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
            )

            data.edge_index = out["edge_index"]
            distances = out["distances"]
        else:
            data.edge_index = radius_graph(
                data.pos, r=self.cutoff, batch=data.batch
            )
            row, col = data.edge_index
            distances = (pos[row] - pos[col]).norm(dim=-1)


        # Allegro works with AtomicDataDicts
        # You need to edit the nequip code to make it work with torch_geometric.data.Data
        
        at_data = AtomicData.to_AtomicDataDict(data)

        at_data['atom_types'] = at_data['atomic_numbers'].long()
        at_data['edge_cell_shift'] = at_data['cell_offsets'].float()

        res = self.model(at_data)

        return res['total_energy'], res['forces']

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy, forces = self._forward(data)

        if self.regress_forces:
            return energy, forces
        else:
            return energy

