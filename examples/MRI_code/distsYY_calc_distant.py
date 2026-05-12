from MRI_data import Brain_dataset_preloaded
from utils import generate_mask
from accuracy_bounds.inverseproblems.feasible_sets_dataloader import distsYY_withineps_dataloader_cuda
from accuracy_bounds.inverseproblems.utils import  torch_csr_to_scipy
from torch.utils.data import DataLoader



import json
import argparse
from scipy import sparse
import os
import multiprocessing




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MRI Kersize computation")

    parser.add_argument('--data_path',type = str ,help='The folder you store your data in, from MRI_data')
    parser.add_argument('--ds_name', type = str, help = 'name of your dataset')
    parser.add_argument('--data_info_path', type = str, help = 'The file where you store the infos about th edataset including the file order and the number of slices, coils per image')
    parser.add_argument('--save_folder',type = str ,help='Where do you want to save your feasibility appartenance and distsXX matrices to disk')
    parser.add_argument('--slice_group', type = int, help = 'The slice group you want to do the computations on')
    parser.add_argument('--acceleration_rate', type = int, help = 'The acceleration rate for the MRI defining the mask')
    parser.add_argument('--SNR', type = int,default = 20 , help = 'Epsilon for the feasible appartenance computation')
    parser.add_argument('--batch_size', type=int, default=3000, help='Batch size to use  in the dataloaders for distance computation')

    #python3 examples/MRI_code/feasapp_calc_distant.py --data_path brain/multicoil_val --ds_name BR_multi --data_info_path  Brain_datainfo_1374.json --save_folder /p/project1/hai_1013/MRI_data/results --slice_group 0 --acceleration_rate 8 --batch_size 1000

    args = parser.parse_args()

    loading = 'preprocessed'

    slice_groups = [[0,1,2], [3,4,5], [5,6,7], [7,8,9], [9,10,11]]

    data_folder = '/p/project1/hai_1013/MRI_data'
    fp_data = f'{data_folder}/{args.data_path}'
    json_path = f'{data_folder}/{args.data_info_path}'
    slices_to_open = slice_groups[args.slice_group]
    save_folder = args.save_folder
    ds_name = args.ds_name
    

    num_cores = multiprocessing.cpu_count()

    
    acceleration_rate = args.acceleration_rate
    SNR = args.SNR
    batchsize_dataloading = args.batch_size


    epsilon_20DB_full = 0.1*0.0227*(8**0.5)
    epsilon_20DB = epsilon_20DB_full/(acceleration_rate**0.5)
    epsilon = epsilon_20DB * 10**((20/20)-(SNR/20))



    with open(json_path, 'r') as f:
        data_info = json.load(f)

    shape_images = (320,320)
    n_central_bands = 22
    mask_subsampling = generate_mask(accel_rate=acceleration_rate, n_central = n_central_bands, shape_k=shape_images)


    dataset_k = Brain_dataset_preloaded(fp_data,data_info, slices_idx= slices_to_open, k_mask= mask_subsampling, type = 'k_mes', shape_x=shape_images)
    dataset_x = Brain_dataset_preloaded(fp_data,data_info, slices_idx= slices_to_open, k_mask= mask_subsampling, type = 'x', shape_x=shape_images)

    dataloader_k =  DataLoader(dataset_k, batch_size=batchsize_dataloading, num_workers = 10)
    dataloader2_k =  DataLoader(dataset_k, batch_size=batchsize_dataloading, num_workers = 10)
    dataloader_x = DataLoader(dataset_x, batch_size=batchsize_dataloading, num_workers = 10)

    
    print('Calculating distsYY')
    distsYY = distsYY_withineps_dataloader_cuda(dataloader_k, dataloader2_k, p_Y = 2, epsilon=epsilon)

    print("Convert to Scipy Sparse")
    distsYY_save = torch_csr_to_scipy(distsYY.cpu().to_sparse_csr())

