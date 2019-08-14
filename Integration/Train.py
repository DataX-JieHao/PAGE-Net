import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from SparseCoding import fixed_s_mask, dropout_mask, sparse_func
from Net import cox_pasnet_pathology
from CostFunc import R_set, neg_par_log_likelihood, c_index

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(x_tr, age_tr, pt_tr, y_tr, delta_tr, \
            x_va, age_va, pt_va, y_va, delta_va, \
            x_te, age_te, pt_te, y_te, delta_te, \
            pathway_indices, \
            gene_nodes, pathway_nodes, image_nodes, hidden_nodes, \
            lr, l2, max_epochs, dropout_rate, step = 100, tolerance = 0.02, sparse_coding = False, test_phrase = False):

    net = cox_pasnet_pathology(gene_nodes, pathway_nodes, image_nodes, hidden_nodes)
    net = net.cuda()
    ###optimizer
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay = l2)
    prev_sum = 0.0
    temp_loss_list = []

    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        net.train()
        opt.zero_grad() ###reset gradients to zeros
        ###Randomize dropout masks
        net.do_m1 = dropout_mask(pathway_nodes, dropout_rate[0])
        net.do_m2 = dropout_mask(hidden_nodes[0], dropout_rate[1])
        net.do_m4 = dropout_mask(image_nodes, dropout_rate[2])
        pred = net(x_tr, age_tr, pt_tr, pathway_indices, dropout_rate) ###Forward
        loss = neg_par_log_likelihood(pred, y_tr, delta_tr) ###calculate loss
        loss.backward() ###calculate gradients
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        net.gene.weight.grad = fixed_s_mask(net.gene.weight.grad, pathway_indices)
        opt.step() ###update weights and biases
        if sparse_coding == True:
            net = sparse_func(net, x_tr, age_tr, pt_tr, y_tr, delta_tr, pathway_indices, dropout_rate)
        torch.cuda.empty_cache()
        net.eval()
        valid_pred = net(x_va, age_va, pt_va, pathway_indices, dropout_rate)
        valid_loss = neg_par_log_likelihood(valid_pred, y_va, delta_va).detach().cpu()
        temp_loss_list.append(valid_loss.detach().cpu())
        if (epoch % step == step - 1) and (epoch + 1 >= 500):
            torch.cuda.empty_cache()
            print("Current LR: ", lr)
            opt_temp_loss = np.min(temp_loss_list)
            gl = valid_loss/opt_temp_loss - 1.0
            net.train()
            pred = net(x_tr, age_tr, pt_tr, pathway_indices, dropout_rate)
            train_cindex = c_index(pred.cpu(), y_tr.cpu(), delta_tr.cpu())
            valid_cindex = c_index(valid_pred.cpu(), y_va.cpu(), delta_va.cpu())
            del pred
            if (gl > tolerance):
                if epoch + 1 == 500: 
                    opt_cidx_tr = train_cindex
                    opt_cidx_ev = valid_cindex
                    opt_net = copy.deepcopy(net)
                print('Early stopping in [%d]' % (epoch + 1))
                print('[%d] GL: %.4f' % (epoch + 1, gl))
                print('[%d] Best CIndex in Train: %.3f' % (epoch + 1, opt_cidx_tr))
                print('[%d] Best CIndex in Valid: %.3f' % (epoch + 1, opt_cidx_ev))
                if (test_phrase == True):
                    opt_net.eval()
                    test_pred = opt_net(x_te, age_te, pt_te, pathway_indices, dropout_rate)
                    opt_cidx_ev = c_index(test_pred.cpu(), y_te.cpu(), delta_te.cpu())
                    print('[%d] Final CIndex in Test: %.3f' % (epoch + 1, opt_cidx_ev))
                break
            else:
                opt_cidx_tr = train_cindex
                opt_cidx_ev = valid_cindex                
                opt_net = copy.deepcopy(net)
                print('[%d] GL: %.4f' % (epoch + 1, gl))
                print('[%d] CIndex in Train: %.3f' % (epoch + 1, train_cindex))
                print('[%d] CIndex in Valid: %.3f' % (epoch + 1, valid_cindex))
                torch.cuda.empty_cache()
                lr = lr*0.8
                if (test_phrase == True):
                    del opt_cidx_ev
                    opt_net.eval()
                    test_pred = opt_net(x_te, age_te, pt_te, pathway_indices, dropout_rate)
                    opt_cidx_ev = c_index(test_pred.cpu(), y_te.cpu(), delta_te.cpu())
                    print('[%d] Final CIndex in Test: %.3f' % (epoch + 1, opt_cidx_ev))
 
    return (opt_cidx_tr, opt_cidx_ev)

def interpret_net(outpath, x, age, pt, y, delta, pathway_indices, \
                    gene_nodes, pathway_nodes, image_nodes, hidden_nodes, \
                    lr, l2, max_epochs, dropout_rate, step = 100, tolerance = 0.05, sparse_coding = False):

    net = cox_pasnet_pathology(gene_nodes, pathway_nodes, image_nodes, hidden_nodes)
    net = net.cuda()
    ###optimizer
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay = l2)
    prev_sum = 0.0
    temp_loss_list = []
    cidx_tr = []
    loss_tr = []
    last_loss = np.inf
    sp_pathway = []
    sp_hidden = []
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        net.train()
        opt.zero_grad() ###reset gradients to zeros
        ###Randomize dropout masks
        net.do_m1 = dropout_mask(pathway_nodes, dropout_rate[0])
        net.do_m2 = dropout_mask(hidden_nodes[0], dropout_rate[1])
        net.do_m4 = dropout_mask(image_nodes, dropout_rate[2])
        pred = net(x, age, pt, pathway_indices, dropout_rate) ###Forward
        loss = neg_par_log_likelihood(pred, y, delta) ###calculate loss
        loss.backward() ###calculate gradients
        ###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
        net.gene.weight.grad = fixed_s_mask(net.gene.weight.grad, pathway_indices)
        opt.step() ###update weights and biases
        if sparse_coding == True:
            net = sparse_func(net, x, age, pt, y, delta, pathway_indices, dropout_rate)
        torch.cuda.empty_cache()
        if epoch % step == step - 1:
            net.eval()
            pred = net(x, age, pt, pathway_indices, dropout_rate)
            train_loss = neg_par_log_likelihood(pred, y, delta).detach().cpu()
            train_cindex = c_index(pred.cpu(), y.cpu(), delta.cpu())
            if train_loss.item() - tolerance > last_loss: 
                break
            else:
                last_loss = train_loss.item()
                opt_net = copy.deepcopy(net)
                loss_tr.append(train_loss.item())
                cidx_tr.append(train_cindex.item())
                sp_pathway.append(net.pathway.weight.nonzero().size(0))
                sp_hidden.append(net.hidden.weight.nonzero().size(0))
                print('[%d] Loss in Train: %.4f' % (epoch + 1, train_loss))
                print('[%d] CIndex in Train: %.3f' % (epoch + 1, train_cindex))
                print('Connections between pathway and hidden layer: ', net.pathway.weight.nonzero().size(0))
                print('Connections between hidden and last hidden layer: ', net.hidden.weight.nonzero().size(0))

    print(step)
    print(epoch)
    x_axis = np.arange(0, epoch + 1, step)
    fig = plt.figure()
    print(len(loss_tr))
    print(len(x_axis))
    plt.plot(x_axis, loss_tr)
    fig.savefig("cindex_"+str(lr)+"_"+str(l2)+".png")
    np.savetxt("sparsity_pathway.txt", sp_pathway, delimiter = ',')
    np.savetxt("sparsity_hidden.txt", sp_hidden, delimiter = ',')
    np.savetxt("cindex_entire_data.txt", cidx_tr, delimiter = ',')
    torch.save(opt_net.state_dict(), outpath)

    return