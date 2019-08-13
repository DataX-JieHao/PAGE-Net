from DataLoader import load_sparse_indices, load_data
from Train import train_model

import torch
import numpy as np
from argparse import ArgumentParser

# parser = ArgumentParser(description="Train gene expression")
# # parser.add_argument("REP_ID", type=int, default=1)
# parser.add_argument("GPU_ID", type=int, default=0)
# parser.add_argument("LR", type=float, default=1e-2)
# parser.add_argument("L2", type=float, default=0.0)

# REPID = parser.parse_args().REP_ID
# GPUID = parser.parse_args().GPU_ID
# lr = parser.parse_args().LR
# l2 = parser.parse_args().L2
'''Set up '''
# torch.cuda.empty_cache()
# torch.cuda.set_device(GPUID)

##### Net
Gene_Nodes = 5404 ### number of genes
Pathway_Nodes = 659 ### number of pathways
Image_Nodes = 50 ### number of feature maps
Hidden_Nodes = [100, 30, 30] ### number of hidden nodes
Max_Epochs = 1500 ###for training
Drop_Rate = [0.7, 0.5, 0.7] ### dropout rates
''' load data '''
folder_path = "/home/NewUsersDir/jhao2/Integrative/src/data/"
pathway_indices = load_sparse_indices(folder_path + "binary_pathway_mask.npz")
C_index = []
for REPID in range(15,20):
    print("-----Split ", REPID)
    x_tr, y_tr, pt_tr, delta_tr, age_tr = load_data(folder_path + "std_train_genomic_" + str(REPID) + ".csv", folder_path + "norm_image_train_" + str(REPID) + ".csv")
    x_va, y_va, pt_va, delta_va, age_va = load_data(folder_path + "std_valid_genomic_" + str(REPID) + ".csv", folder_path + "norm_image_valid_" + str(REPID) + ".csv")
    x_te, y_te, pt_te, delta_te, age_te = load_data(folder_path + "std_test_genomic_" + str(REPID) + ".csv", folder_path + "norm_image_test_" + str(REPID) + ".csv")
    ###grid search the optimal hyperparameters using train and validation data
    # L2_Lambda = [0.01, 0.02, 0.04, 0.08, 0.0016]
    L2_Lambda = [0.1,0.2,0.3,0.4,0.5]
    # L2_Lambda = [0.2, 0.4,0.5]
    # L2_Lambda = [0.08]
    # L2_Lambda = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # Initial_Learning_Rate = [0.00]
    # Initial_Learning_Rate = [0.005]
    # Initial_Learning_Rate = [0.0015]
    Initial_Learning_Rate = [0.001,0.0015,0.0001,0.00015]
    # Initial_Learning_Rate = [0.0008]
    # Initial_Learning_Rate = [0.002]
    # Initial_Learning_Rate = [0.001, 0.0002, 0.00004]
    # Initial_Learning_Rate = [0.1, 0.02, 0.004, 0.008]
    # Initial_Learning_Rate = [0.001, 0.0001, 0.00001, 0.000001]
    # L2_Lambda = [0.005, 0.01, 0.02, 0.04, 0.08, 0.10]
    opt_cidx = 0.0
    for lr in Initial_Learning_Rate:
        for l2 in L2_Lambda:
        # for lr in Initial_Learning_Rate:
            # print("--------------------")
            # print("No sparse coding")
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