import numpy as np
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import json
from utils import (
    dft2, generate_mask, inverse_fft2_shift, ImgComparator, ManualPolygonDrawer,
    apply_polygon_mask, kernel_projection, kernel_draw, transform_batch,
    predict_batch, load_model
)
from functools import partial
import torch

def main():
    """
    Main script for MRI data processing, visualization, and prediction.
    This script loads MRI slices, applies masks, allows interactive ROI selection,
    and optionally runs predictions using a pre-trained model.
    """
    # --- Configuration ---
    slice_group = 0
    acceleration_rate = 8
    SNR = 50
    shape_k = (320, 320)
    epsilon_20DB_full = 0.1 * 0.0227 * (8**0.5)  # Value corresponding to a common l2 norm of fully sampled MRI Scan in the processed Brain SC data
    epsilon_20DB = epsilon_20DB_full / (8**0.5)
    epsilon = epsilon_20DB * 10**((20/20) - (SNR/20))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # --- Generate mask and projection functions ---
    k_mask = generate_mask(acceleration_rate, 22, shape_k)
    P_null = partial(kernel_projection, k_mask=k_mask, max_norm=epsilon)
    kerdraw = partial(kernel_draw, P_null=P_null)

    # --- Data paths ---
    data_folder = '/localhome/iaga_dv/Dokumente/MRI_data'
    polygon_path = f'{data_folder}/brain/ROI_null_test.json'
    subds_folder = '/localhome/iaga_dv/Dokumente/MRI_data/brain/SC_processed_subds_test'
    filenames = os.listdir(subds_folder)
    scans_ids = list(set([x.split('_')[0] for x in filenames]))

    # --- Main loop over scans ---
    for scan_id in tqdm(scans_ids):
        # --- Load data ---
        path_sl0 = os.path.join(subds_folder, f'{scan_id}_sl0.npy')
        path_sl1 = os.path.join(subds_folder, f'{scan_id}_sl1.npy')
        GT_img = np.load(path_sl0)
        lim1_img = np.load(path_sl1)

        # --- Transform to k-space ---
        lim1_kfull = dft2(lim1_img)
        GT_kfull = dft2(GT_img)

        # --- Apply mask ---
        lim1_ksub = lim1_kfull * k_mask
        input_img = GT_kfull * k_mask

        # --- Compute k-space distance ---
        k_dist = np.linalg.norm(lim1_ksub - input_img, ord=2)
        y_0 = (lim1_ksub + input_img) / 2

        # --- Transform back to image space ---
        lim1_xsub = inverse_fft2_shift(lim1_ksub)
        y_0_xsub = inverse_fft2_shift(y_0)
        input_xsub = inverse_fft2_shift(input_img)

        # --- Visualize base and detail slices ---
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        axes[0].imshow(np.abs(GT_img))
        axes[0].set_title('Slice 0 scan (base scan)')
        axes[1].imshow(np.abs(lim1_img))
        axes[1].set_title('Slice 1 scan (to take the detail from)')
        fig.suptitle(
            'Commands : \n'
            'y for adding a point in the current polygon \n'
            'c to cancel the current polygon \n'
            'd to delete the last closed polygon \n'
            '0 to close the current polygon and start a new one \n'
            'ENTER to finish selecting details to paste'
        )
        comp = ImgComparator(fig, axes)
        drawer = ManualPolygonDrawer(fig, slice_group, polygon_path)
        plt.tight_layout()
        plt.show()

        # --- Load and apply ROI polygons ---
        with open(polygon_path, 'r') as f:
            polygons = json.load(f)[str(slice_group)]
        for polygon in polygons:
            mask_ROI = 1 - apply_polygon_mask(np.ones((320, 320)), [polygon])

        # --- User input for detail amplification ---
        scale = float(input('How much do you want to amplify the detail before projection and pasting? (Neutral: 1, Usually between 1 and 2) '))
        lim1_artificial = kerdraw(GT_img, lim1_img, mask_ROI, scale=scale)

        # --- Visualize results ---
        k_lim1_artificial = k_mask * dft2(lim1_artificial)
        x_sub_artificial = inverse_fft2_shift(k_lim1_artificial)
        x_sub_delta = inverse_fft2_shift(k_lim1_artificial - input_img)

        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        axes[0].imshow(np.abs(input_xsub))
        axes[0].set_title('Original input')
        axes[1].imshow(np.abs(x_sub_artificial))
        axes[1].set_title('Subsampled modified img')
        axes[2].imshow(np.abs(x_sub_delta))
        axes[2].set_title('Difference of subsampled')

        noise_ratio = np.linalg.norm(x_sub_delta, ord=2) / np.linalg.norm(x_sub_artificial, ord=2)
        SNR_val = -20 * np.log10(noise_ratio)
        fig.suptitle(f'SNR = {SNR_val:.2f}')

        axes[3].imshow(GT_img)
        axes[3].set_title('Original GT')
        axes[4].imshow(np.abs(lim1_artificial))
        axes[4].set_title('Modified img')
        im = axes[5].imshow(np.abs(GT_img - lim1_artificial))
        axes[5].set_title('Delta in x')
        plt.tight_layout()
        comp = ImgComparator(fig, axes)
        plt.show()

        # --- Optional prediction ---
        prediction = input('Do you want to predict this one? (y/n) ')
        if 'y' in prediction.lower():
            k_SC = np.stack([input_img, k_lim1_artificial], axis=0)
            # --- Apply transform to get an input batch ---
            k_trans, means, stds = transform_batch(k_SC, k_mask)
            # --- Apply model on transformed batch ---
            preds_batch = predict_batch(model, k_trans, means, stds, device)

            pred_fromy = preds_batch[0]
            pred_fromartificial = preds_batch[1]
            forwarded_pred_fromy = dft2(pred_fromy)

            # --- Visualize predictions ---
            fig, axes = plt.subplots(2, 3)
            axes = axes.flatten()
            axes[0].imshow(GT_img)
            axes[0].set_title('Original GT')
            axes[1].imshow(np.abs(lim1_artificial))
            axes[1].set_title('Modified img')
            im = axes[2].imshow(np.abs(GT_img - lim1_artificial))
            axes[2].set_title('Delta in x')
            fig.colorbar(im, ax=axes[2])

            axes[3].imshow(pred_fromy)
            axes[3].set_title('Pred from original input')
            axes[4].imshow(pred_fromartificial)
            axes[4].set_title('Pred from modified img')
            im = axes[5].imshow(np.abs(pred_fromy - pred_fromartificial))
            axes[5].set_title('Delta pred')
            fig.colorbar(im, ax=axes[5])
            plt.tight_layout()
            comp = ImgComparator(fig, axes)
            plt.show()

            # --- Show individual images ---
            plt.imshow(np.abs(GT_img), cmap='gray')
            plt.title('x (base image)')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            plt.imshow(np.abs(lim1_artificial), cmap='gray')
            plt.title('x + x_det')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            plt.imshow(np.abs(lim1_img), cmap='gray')
            plt.title('Image where x_det is originally from')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            plt.imshow(np.abs(pred_fromy), cmap='gray')
            plt.title('Pred from GT')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            plt.imshow(np.abs(pred_fromartificial), cmap='gray')
            plt.title('Pred from artificial')
            plt.xticks([])
            plt.yticks([])
            plt.show()

if __name__ == '__main__':
    main()
        
