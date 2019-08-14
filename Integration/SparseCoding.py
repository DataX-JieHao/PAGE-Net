import torch 
import numpy as np
import math
import copy
from CostFunc import R_set, neg_par_log_likelihood, c_index
from scipy.interpolate import interp1d

def fixed_s_mask(w, idx):

    sp_w = torch.sparse_coo_tensor(idx, w[idx], w.size())
    return(sp_w.to_dense())



def dropout_mask(n_node, drop_p):
    '''Construct a binary matrix to randomly drop nodes in a layer.
    Input:
        n_node: number of nodes in the layer.
        drop_p: the probability that a node is to be dropped.
    Output:
        mask: a binary matrix, where 1 --> keep the node; 0 --> drop the node.
    '''

    keep_p = 1.0 - drop_p
    mask = torch.Tensor(np.random.binomial(1, keep_p, size=n_node))
    ###if gpu is being used
    if torch.cuda.is_available():
        mask = mask.cuda()
    ###
    return(mask)



def small_net_mask(w, m_in_nodes, m_out_nodes):

    nonzero_idx_in = m_in_nodes.nonzero()
    nonzero_idx_out = m_out_nodes.nonzero()
    sparse_row_idx = nonzero_idx_out.repeat(nonzero_idx_in.size()).transpose(1,-2)
    sparse_col_idx = nonzero_idx_in.repeat(nonzero_idx_out.size()).transpose(1,-2)
    idx = torch.cat((sparse_row_idx, sparse_col_idx), 0)
    val = torch.ones(nonzero_idx_out.size(0)*nonzero_idx_in.size(0))
    sparse_bool_mask = torch.sparse_coo_tensor(idx, val, w.size())
    ##if gpu is being used
    if torch.cuda.is_available():
        sparse_bool_mask = sparse_bool_mask.cuda()
    ###
    mask = sparse_bool_mask.to_dense()
    return(mask.type(torch.uint8))



def soft_threshold(w, th):
    '''Soft-thresholding'''
    return torch.sign(w)*torch.clamp(abs(w) - th, min=0.0)



def get_threshold(w, m, sp):

    pos_param = torch.abs(torch.masked_select(w, m))
    ###obtain the kth number based on sparse_level
    top_k = math.ceil(pos_param.size(0) * (100 - sp) * 0.01)
    return(torch.topk(pos_param, top_k)[0][-1])



def get_best_sparsity(sparse_set, loss_set):

    interp_loss_set = interp1d(sparse_set, loss_set, kind='cubic')
    interp_sparse_set = torch.linspace(min(sparse_set), max(sparse_set), steps=100)
    interp_loss = interp_loss_set(interp_sparse_set)
    best_sp = interp_sparse_set[np.argmin(interp_loss)]
    return(best_sp)



def get_sparse_weight(w, m, s):

    epsilon = get_threshold(w, m, s)
    sp_w = soft_threshold(w, epsilon)
    return(sp_w)



def sparse_func(net, train_x, train_age, train_pt, train_ytime, train_yevent, Pathway_Indices, Dropout_Rate):

    ###serializing net 
    net_state_dict = net.state_dict()
    ###make a copy for net, and then optimize sparsity level via copied net
    copy_net = copy.deepcopy(net)
    copy_state_dict = copy_net.state_dict()

    for name, param in net_state_dict.items():
        ###omit the param if it is not a weight matrix
        if not "weight" in name: continue
        if "omics" in name: continue
        if "gene" in name: continue
        if "integrative" in name: continue
        if "bn1" in name: continue
        if "bn2" in name: continue
        if "bn3" in name: continue
        if "bn4" in name: continue
        if "pathway" in name:
            active_mask = small_net_mask(net.pathway.weight.data, net.do_m1, net.do_m2)
            copy_weight = copy.deepcopy(net.pathway.weight.data)
        if "hidden" in name:
            active_mask = small_net_mask(net.hidden.weight.data, net.do_m2, net.do_m3)
            copy_weight = copy.deepcopy(net.hidden.weight.data)
        if "image" in name:
            active_mask = small_net_mask(net.image.weight.data, net.do_m4, net.do_m5)
            copy_weight = copy.deepcopy(net.image.weight.data)
        S_set = torch.linspace(99, 0, 5) 
        S_loss = []
        for S in S_set:
            sp_param = get_sparse_weight(copy_weight, active_mask, S.item())
            copy_state_dict[name].copy_(sp_param)
            copy_net.train()
            y_tmp = copy_net(train_x, train_age, train_pt, Pathway_Indices, Dropout_Rate)
            loss_tmp = neg_par_log_likelihood(y_tmp, train_ytime, train_yevent)
            S_loss.append(loss_tmp)
        ###apply cubic interpolation
        best_S = get_best_sparsity(S_set, S_loss)
        best_epsilon = get_threshold(copy_weight, active_mask, best_S)
        optimal_sp_param = soft_threshold(copy_weight, best_epsilon)
        copy_weight[active_mask] = optimal_sp_param[active_mask]
        ###update weights in copied net
        copy_state_dict[name].copy_(copy_weight)
        ###update weights in net
        net_state_dict[name].copy_(copy_weight)
    
    return(net)