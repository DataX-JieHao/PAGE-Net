from DataLoader import load_sparse_indices, load_data
from Train import train_model

import torch
import numpy as np

##### Net Settings #####
Gene_Nodes = 5404 ### number of genes
Pathway_Nodes = 659 ### number of pathways
Image_Nodes = 50 ### number of aggregated feature maps
Hidden_Nodes = [100, 30, 30] ### number of hidden nodes in hidden, hidden 2, and hidden 3
Max_Epochs = 1500 ### maximum number of epochs in training
### sub-network setup
Drop_Rate = [0.7, 0.5, 0.7] ### dropout rates of pathway, hidden, image layers

''' load data and pathway'''
folder_path = "/home/NewUsersDir/jhao2/Integrative/src/data/"
pathway_indices = load_sparse_indices(folder_path + "binary_pathway_mask.npz")

C_index = []

for REPID in range(20):
    print("-----Split ", REPID)
    x_tr, y_tr, pt_tr, delta_tr, age_tr = load_data(folder_path + "std_train_genomic_" + str(REPID) + ".csv", folder_path + "norm_image_train_" + str(REPID) + ".csv")
    x_va, y_va, pt_va, delta_va, age_va = load_data(folder_path + "std_valid_genomic_" + str(REPID) + ".csv", folder_path + "norm_image_valid_" + str(REPID) + ".csv")
    x_te, y_te, pt_te, delta_te, age_te = load_data(folder_path + "std_test_genomic_" + str(REPID) + ".csv", folder_path + "norm_image_test_" + str(REPID) + ".csv")
    ###grid search the optimal hyperparameters using train and validation data
     L2_Lambda = [0.1,0.2,0.3,0.4,0.5]
    Initial_Learning_Rate = [0.001,0.0015,0.0001,0.00015]
    opt_cidx = 0.0
    for lr in Initial_Learning_Rate:
        for l2 in L2_Lambda:
            print("Sparse coding is on")
            print("L2: ", l2, "LR: ", lr)
            torch.cuda.empty_cache()
            tr_cindex, va_cindex = train_model(x_tr, age_tr, pt_tr, y_tr, delta_tr, \
                x_va, age_va, pt_va, y_va, delta_va, x_te, age_te, pt_te, y_te, delta_te, \
                pathway_indices, \
                Gene_Nodes, Pathway_Nodes, Image_Nodes, Hidden_Nodes, \
                lr, l2, Max_Epochs, Drop_Rate, step = 100, tolerance = 0.15, \
                sparse_coding = True, test_phrase = False)
            if (tr_cindex.item() > va_cindex.item()) and (va_cindex.item() > opt_cidx):
                opt_l2 = l2
                opt_lr = lr
                opt_cidx_tr = tr_cindex
                opt_cidx = va_cindex
            torch.cuda.empty_cache()
    print("--------------------")
    print("Optimal l2: ", opt_l2, "Optimal lr: ", opt_lr)
    print("Optimal C-Index in Train: ", opt_cidx_tr, "Optimal C-Index in Valid: ", opt_cidx)
    print("--------------------")

    print("---Maximize c-index---")
    tr_cindex, te_cindex = train_model(x_tr, age_tr, pt_tr, y_tr, delta_tr, \
        x_va, age_va, pt_va, y_va, delta_va, x_te, age_te, pt_te, y_te, delta_te, \
        pathway_indices, \
        Gene_Nodes, Pathway_Nodes, Image_Nodes, Hidden_Nodes, \
        opt_lr, opt_l2, Max_Epochs, Drop_Rate, step = 100, tolerance = 0.5, \
        sparse_coding = True, test_phrase = True)
    print("PagenetC-Index in Train: ", tr_cindex.item())
    print("PagenetC-Index in Test: ", te_cindex.item())
    print("....................")
    C_index.append(te_cindex.item())

np.savetxt("integration_" + str(REPID) + ".txt", np.array(C_index))