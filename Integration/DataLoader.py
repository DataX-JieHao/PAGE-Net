import numpy as np
import pandas as pd
from scipy import sparse
import torch

def load_sparse_indices(path):
    '''Load the sparse format's binary matrix.
    Input:
        path: path to input dataset in sparse format
    Output:
        indices: nonzero indices
    '''

    coo = sparse.load_npz(path)
    indices = np.vstack((coo.row, coo.col))

    return(indices)



def sort_data(path):
    ''' sort the genomic and clinical data w.r.t. survival time (OS_MONTHS) in descending order
    Input:
        path: path to input dataset (which is expected to be a csv file)
    Output:
        sort_id: the sorted indices of 'SAMPLE_ID'
        x: sorted genomic inputs
        y: sorted survival time (OS_MONTHS) corresponding to 'x'
        delta: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored
        age: sorted age corresponding to 'x'
    '''

    data = pd.read_csv(path)

    data.sort_values("OS_MONTHS", ascending = False, inplace = True)
    data = data.reset_index(drop = True)
    
    sort_id = data["SAMPLE_ID"]
    x = data.drop(["SAMPLE_ID", "OS_MONTHS", "OS_EVENT", "AGE"], axis = 1).values
    y = data.loc[:, ["OS_MONTHS"]].values
    delta = data.loc[:, ["OS_EVENT"]].values
    age = data.loc[:, ["AGE"]].values

    return(sort_id, x, y, delta, age)



def load_data(input_path, feature_map):
    '''load the sorted indices of 'SAMPLE_ID' and aggregated feature maps from the model trained by pathological images
    Input:
        input_path: path to genomic and clinical input dataset
        feature_map: path to aggregated image features
    Output:
        x: sorted genomic inputs
        y: sorted survival time (OS_MONTHS) corresponding to 'x'
        pt: sorted pathologcial aggregated inputs
        delta: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored
        age: sorted age corresponding to 'x'
    '''

    sort_id, x, y, delta, age = sort_data(input_path)

    x = torch.from_numpy(x).to(dtype=torch.float).cuda()
    y = torch.from_numpy(y).to(dtype=torch.float).cuda()
    delta = torch.from_numpy(delta).to(dtype=torch.float).cuda()
    age = torch.from_numpy(age).to(dtype=torch.float).cuda()

    fm = pd.read_csv(feature_map)
    fm = fm.set_index("SAMPLE_ID")
    fm = fm.reindex(index = sort_id).values
    pt = torch.from_numpy(fm).to(dtype=torch.float).cuda()

    return(x, y, pt, delta, age)
