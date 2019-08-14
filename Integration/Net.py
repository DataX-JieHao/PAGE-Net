from SparseCoding import fixed_s_mask
import torch
import torch.nn as nn

class cox_pasnet_pathology(nn.Module):
    ''' Genomic: Two hidden layers
        Clinical: One clinical layer (age)
        Pathological: One hidden layer
    '''
    def __init__(self, gene_nodes, pathway_nodes, image_nodes, hidden_nodes):
        super(cox_pasnet_pathology, self).__init__()
        # self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        ###gene layer --> pathway layer
        self.gene = nn.Linear(gene_nodes, pathway_nodes)
        ###pathway layer --> hidden layer
        self.pathway = nn.Linear(pathway_nodes, hidden_nodes[0])
        ###hidden layer --> hidden 2 layer
        self.hidden = nn.Linear(hidden_nodes[0], hidden_nodes[1])
        ###image layer --> hidden 3 layer
        self.image = nn.Linear(image_nodes, hidden_nodes[2])
        ###hidden 2 + hidden 3 + age --> Cox layer
        self.integrative = nn.Linear(hidden_nodes[1] + hidden_nodes[2] + 1, 1, bias = False)
        ###batch normalization
        self.bn1 = nn.BatchNorm1d(pathway_nodes)
        self.bn2 = nn.BatchNorm1d(hidden_nodes[0])
        self.bn3 = nn.BatchNorm1d(hidden_nodes[1])
        self.bn4 = nn.BatchNorm1d(hidden_nodes[2])
        ###randomly select a small sub-network
        self.do_m1 = torch.ones(pathway_nodes).cuda()
        self.do_m2 = torch.ones(hidden_nodes[0]).cuda()
        self.do_m3 = torch.ones(hidden_nodes[1]).cuda()
        self.do_m4 = torch.ones(image_nodes).cuda()
        self.do_m5 = torch.ones(hidden_nodes[2]).cuda()

    def forward(self, x_1, x_2, x_3, pathway_idx, Drop_Rate):
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        self.gene.weight.data = fixed_s_mask(self.gene.weight.data, pathway_idx)

        x_1 = self.tanh(self.bn1(self.gene(x_1)))

        if self.training == True: 
            x_1 = (1/(1-Drop_Rate[0])) * x_1.mul(self.do_m1)

        x_1 = self.tanh(self.bn2(self.pathway(x_1)))

        if self.training == True: 
            x_1 = (1 / (1 - Drop_Rate[1])) * x_1.mul(self.do_m2)

        x_1 = self.tanh(self.bn3(self.hidden(x_1)))

        if self.training == True: 
            x_3 = (1 / (1 - Drop_Rate[2])) * x_3.mul(self.do_m4)
        
        x_3 = self.tanh(self.bn4(self.image(x_3)))

        ###integration
        x_cat = torch.cat((x_1, x_2, x_3), 1)
        lin_pred = self.integrative(x_cat)
        
        return lin_pred

