import json
import os
from matplotlib import pyplot as plt
import torch
import h5py
import numpy as np
from functools import partial
from scipy.ndimage import binary_dilation
from tqdm import tqdm
from utils import load_model, ImgPasting, generate_mask, kernel_draw, kernel_projection, ImgComparator, transform_batch, predict_batch, dft2, inverse_fft2_shift




if __name__ == '__main__':

    data_folder = '/localhome/iaga_dv/Dokumente/MRI_data'
    processed_SC_folder_brain = f'{data_folder}/brain/SC_processed_test'
    annotation_version = 'v0'
    annotation_path = f'{data_folder}/annotation_info_{annotation_version}.json'
    anomalous_folder = f'{data_folder}/brain/anomalous_slices_v0/slices'
    anomalous_ROI_folder = f'{data_folder}/brain/anomalous_slices_v0/ROI_annotations'

    acceleration_rate = 8
    shape_k = (320,320)
    SNR = 50
    epsilon_20DB_full = 0.1*0.0227*(8**0.5) # Value corresponding to a common l2 norm of fully sampled MRI Scan in the processed Brain SC data
    epsilon_20DB = epsilon_20DB_full/(8**0.5)
    epsilon = epsilon_20DB * 10**((20/20)-(SNR/20))

    k_mask = generate_mask(acceleration_rate, 22,shape_k)
    P_null = partial(kernel_projection, k_mask = k_mask, max_norm = epsilon)
    kerdraw = partial(kernel_draw,P_null = P_null )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = load_model(device)


    annotation_version = 'v0'
    annotation_path = f'{data_folder}/annotation_info_{annotation_version}.json'
    with open(annotation_path, 'r') as f:
        annotation_info = json.load(f)

    association = annotation_info['association']
    slicelist = annotation_info['slicelist']

    filenames = os.listdir(processed_SC_folder_brain)
    for file in filenames:
        with h5py.File(os.path.join(processed_SC_folder_brain, file), "r") as f:
            GT_img = f['x'][()][0]


        q99 = np.quantile(GT_img, 0.99)
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(np.clip(np.flipud(GT_img), 0, q99), cmap = 'gray', origin = 'upper')
        axes[0].set_xlim(0, 320)
        axes[0].set_ylim(0, 320)

        axes[1].imshow(np.zeros((320,320)))
        axes[1].set_title('Anomaly in the original scan')

        fig.suptitle('Commands : \n Use arrows to navigate through anomalies \n Right click to place the anomaly somewhere \n + and - to adjust the size of the anomaly \n ENTER to validate the pasting position and size')

        # Select an anomaly type in the console
        # Switch between different anomalies of this type with the keybo
        ip = ImgPasting(fig, anomalies_assoc=association, slicelist=slicelist, folder_scans=anomalous_folder, folder_ROI=anomalous_ROI_folder)
        plt.show()

        new_detail = ip.new_detail
        original_anomalous = ip.anomalous_slice

        mask_ROI = (new_detail>0).astype(np.uint8)
        
        scale = float(input('How much do you want to amplify the detail before projection and pasting ? (Neutral : 1 , Usually between 1 and 2)'))
        artificial_img = kerdraw(GT_img, new_detail, mask_ROI, scale=scale)

  
        fig, axes = plt.subplots(1,3)
        q99 = np.quantile(GT_img, 0.99)
        axes[0].imshow(np.clip(GT_img, 0, q99))
        axes[0].set_title('GT image')

        q99 = np.quantile(artificial_img, 0.99)
        axes[1].imshow(np.clip(artificial_img, 0, q99))
        axes[1].set_title('Artificial image')

        q99 = np.quantile(original_anomalous, 0.99)
        axes[2].imshow(np.clip(original_anomalous, 0, q99))
        axes[2].set_title('Original anomalous slice')

        comp = ImgComparator(fig, axes)
        plt.show()

            
        k_lim1_artificial = k_mask * dft2(artificial_img)
        input_img = k_mask* dft2(GT_img)

        input_xsub = inverse_fft2_shift(input_img)
        x_sub_artificial = inverse_fft2_shift(k_lim1_artificial)
        x_sub_delta = inverse_fft2_shift(k_lim1_artificial- input_img)

        fig, axes = plt.subplots(1,3)
        axes[0].imshow(np.abs(input_xsub))
        axes[0].set_title('original input')

        axes[1].imshow(np.abs(x_sub_artificial))
        axes[1].set_title('Subsampled modified img')

        axes[2].imshow(np.abs(x_sub_delta))
        axes[2].set_title('Difference of subsampled')

        noise_ratio = np.linalg.norm(x_sub_delta, ord = 2)/np.linalg.norm(x_sub_artificial, ord = 2)
        SNR_val = -20*np.log10(noise_ratio)
        fig.suptitle(f'SNR = {SNR_val} ')

        plt.tight_layout()
        comp = ImgComparator(fig, axes)
        plt.show()
        predict_choice = input('Do you want to predict that sample ? (y/n)')
        if 'y' in predict_choice:

            k_SC = np.stack([input_img, k_lim1_artificial], axis = 0)
            # Apply transform to get an input batch
            k_trans, means, stds = transform_batch(k_SC, k_mask)
            # apply model on transformed batch
            preds_batch = predict_batch(model, k_trans, means, stds, device)

            pred_fromy = preds_batch[0]
            pred_fromartificial = preds_batch[1]

            forwarded_pred_fromy = k_mask*dft2(pred_fromy)
            forwarded_pred_fromartificial = k_mask*dft2(pred_fromartificial)
            
            x_sub_pred_from_artificial = inverse_fft2_shift(forwarded_pred_fromartificial)
            x_sub_pred_fromy = inverse_fft2_shift(forwarded_pred_fromy)

            fig, axes = plt.subplots(2,3)
            axes = axes.flatten()

            q99 = np.quantile(np.abs(GT_img), 0.99)
            axes[0].imshow(np.clip(np.abs(GT_img), 0, q99))
            axes[0].set_title('original GT')


            q99 = np.quantile(np.abs(artificial_img), 0.99)
            axes[1].imshow(np.clip(np.abs(artificial_img), 0, q99))
            axes[1].set_title('Modified img')


            im = axes[2].imshow(np.abs(GT_img- artificial_img))
            axes[2].set_title('Delta in x')
            fig.colorbar(im, ax=axes[2])


            axes[3].imshow(pred_fromy)
            axes[3].set_title('Pred from original input')

            axes[4].imshow(pred_fromartificial)
            axes[4].set_title('Pred from modified img')

            im = axes[5].imshow(np.abs(pred_fromy-pred_fromartificial))
            axes[5].set_title('delta pred ')
            fig.colorbar(im, ax=axes[5])

    
            plt.tight_layout()
            comp = ImgComparator(fig, axes)
            plt.show()

            q99 = np.quantile(np.abs(GT_img), 0.99)
            plt.imshow(np.clip(np.abs(GT_img), 0, q99), cmap = 'gray')
            print('x (base image)')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            q99 = np.quantile(np.abs(artificial_img), 0.99)
            plt.imshow(np.clip(np.abs(artificial_img),0, q99), cmap = 'gray')
            print('x + x_det')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            q99 = np.quantile(np.abs(original_anomalous), 0.99)
            plt.imshow(np.clip(np.abs(original_anomalous), 0, q99), cmap = 'gray')
            print('image where x_det is originally from')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            q99 = np.quantile(np.abs(pred_fromy), 0.99)
            plt.imshow(np.clip(np.abs(pred_fromy), 0,q99), cmap = 'gray')
            print('Pred from GT')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            q99 = np.quantile(np.abs(pred_fromartificial), 0.99)
            plt.imshow(np.clip(np.abs(pred_fromartificial), 0, q99), cmap = 'gray')
            print('Pred from artificial')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            pred_noisy = input('Do you want to  proceed to the hallucination definition verification for that sample (adding noise 10 times and predicting might take longer)? (y/n)')
            if 'y' in pred_noisy:
                p_X = 1
                n_preds = 10
                structure = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]], dtype=int)
                ROI_norm = binary_dilation(mask_ROI, structure=structure, iterations=3)
                plt.imshow(ROI_norm)
                plt.axis('off')
                print('ROI for norm')
                plt.show()

                # record the norm of x_det on that ROI
                x_det_norm = np.linalg.norm((artificial_img-pred_fromartificial)*mask_ROI, ord = p_X)

                # Set epsilon for noise adding
                N_pix = k_mask.sum()
                sigma = epsilon/(N_pix**0.5)
            
                # 10 times : 
                preds_batch = []
                for i in tqdm(range(n_preds)):

                #   Add noise in the y space (verz small, like 50 db)
                    noise_vec = np.random.normal(loc=0.0, scale= sigma, size=(320, 320))*k_mask # In fourrier space
                    input_noisy = k_lim1_artificial + noise_vec
                    #   Pred from that noise
                    k_SC = np.stack([input_noisy], axis = 0)
                    k_trans, means, stds = transform_batch(k_SC, k_mask)
                    pred_batch = predict_batch(model, k_trans, means, stds, device)[0]
                    preds_batch.append(pred_batch)
                preds_batch = np.stack(preds_batch, axis = 0)

                #   Calc the dist to x+x_det and to x
                
                x_det_repeated = np.repeat(GT_img[None, :, :], n_preds, axis=0)  # x+x_det
                #x_det_repeated = np.repeat(pred_fromartificial[None, :, :], n_preds, axis=0)  # x+x_det
                artificial_repeated = np.repeat(artificial_img[None, :, :], n_preds, axis=0) # x

                dist_to_x = np.linalg.norm( (preds_batch- artificial_repeated)*mask_ROI[None,:,:], ord = p_X, axis = (1,2))
                dist_to_x_det = np.linalg.norm( (preds_batch- x_det_repeated)*mask_ROI[None,:,:], ord = p_X, axis = (1,2))

                #   Exlude the pred of the dist to x is smaller than x_det/2
                valid_indices = dist_to_x_det < (0.5* x_det_norm)
                dists_to_xdet_noisy = dist_to_x_det[valid_indices]
                dists_to_x_noisy = dist_to_x[valid_indices]

                # Record the number of noise vectors
                print(f'Number of valid preds from noisy inputs : {dists_to_xdet_noisy.shape[0]} / {n_preds} ')
                # Record eta min, eta max
                if dists_to_xdet_noisy.shape[0]>0:
                    eta_min = dists_to_xdet_noisy.max()
                    eta_max = dists_to_x_noisy.min()
                    print(f'For preds from noisy inputs : eta_min = {eta_min} , eta_max = {eta_max} ')




    