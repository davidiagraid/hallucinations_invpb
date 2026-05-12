from matplotlib import pyplot as plt
from utils import ImgPastingS2, apply_upsampling, get_distance, ImgComparator, find_lr_data_idx, save_into_tiff, apply_square_op_small
import rasterio
import os
import torch
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from functools import partial


def kernel_draw(base_img, lim1_img, mask_ROI, P_null, scale = 1, sigma_blend = 1):
    detail = mask_ROI * (scale*lim1_img-base_img)
    xs,ys, bands = np.where(mask_ROI==1)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    x_center = (xmin + xmax)//2
    y_center = (ymin + ymax)//2

    detail_padded = np.pad(detail, ((64, 64), (64, 64), (0, 0)),mode='constant', constant_values=0)
    # Get a patch of shape 128,128 surrounding the detail. Outside the detail, it is made of zeros
    detail_patch = detail_padded[x_center: x_center + 128, y_center:y_center+128, :]

    # Project it onto the null space
    detail_proj = np.array(P_null(img = detail_patch.transpose(2,0,1)).permute(1,2,0))

    # Paste  the base + delta at the right apot + using alpha blending
    detail_padded[x_center: x_center + 128, y_center:y_center+128, :] = detail_proj
    detail_proj = detail_padded[64:-64, 64:-64,:]

    # alpha blend the delta proj + GT_img on GT_img
    lim1_artificial = alpha_blending(base_img, mask_ROI.astype(np.uint8), detail_proj, sigma_blend=sigma_blend)

    return lim1_artificial

def alpha_blending(img_origin,mask_ROI, detail, sigma_blend):

    detail+= img_origin
    
    # Erode the blurred mask by 1 pixel
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask_ROI, kernel, iterations=1).astype(float)
    # Gaussian convolution of the mask 
    blurred_mask = gaussian_filter(eroded_mask, sigma = sigma_blend)

    res_noblend = np.copy(img_origin)
    res_noblend[mask_ROI == 1] = detail[mask_ROI == 1]

    res_blended = blurred_mask * res_noblend + (1-blurred_mask)* img_origin
    
    return res_blended


if __name__ == '__main__':
    folder_drawing_pairs = '/localhome/iaga_dv/Dokumente/sat_data/CP_drawing_pairs'
    dataset_folder = '/localhome/iaga_dv/Dokumente/sat_data/cross_processed'

    subdatasets = ['naip', 'spain_crops', 'spain_urban', 'spot']
    image_ids = []
    for subds in subdatasets:
        images_id_subds = os.listdir(os.path.join(dataset_folder, subds))
        images_id_subds = [(subds, x) for x in images_id_subds]
        for x in images_id_subds:
            image_ids.append(x)
    #print(image_ids)
    img_ids = [x for x in image_ids if 'json' not in x[1]]

    P_null_mat = np.load('../Operators/P_null_32.npy')
    P_null = partial(apply_square_op_small, Op_Mat = P_null_mat, out_2Dshape = (128,128))


    fig, axes = plt.subplots(1,2)
    fig.suptitle('Commands : \nUse arrows to navigate between the images in the dataset\n y for adding a point in the current polygon \n c to cancel the current polygon \n d to delete the last closed polygon \n 0 to close the current polygon and start a new one \n Right click to place the drawing \n ENTER to finish selecting details to paste')

    img_paster = ImgPastingS2(fig, img_ids, dataset_folder)
    plt.show()

    new_detail = img_paster.new_detail
    base_image = img_paster.img_base.get_array()
    base_img_id = img_paster.base_img_id

    mask_ROI = new_detail>0

    scale = float(input('Enter a scale factor for the kernel drawing \n'))
    artificial_img = kernel_draw(base_image, new_detail, mask_ROI=mask_ROI, P_null=P_null, scale = scale)

    base_image = torch.tensor(base_image.transpose(2,0,1)*3000)
    artificial_img  =torch.tensor(artificial_img.transpose(2,0,1)*3000)
    delta_HR = get_distance(base_image, artificial_img, method='l1', agg_method='pixel')

    artificial_LR = apply_upsampling(artificial_img, scale = 4)
    base_LR = apply_upsampling(base_image, scale = 4)
    delta_LR = get_distance(base_LR, artificial_LR, method='l1', agg_method='pixel')

    fig, axes = plt.subplots(1,3)

    axes[0].imshow(base_image.permute(1,2,0)/3000)
    axes[0].set_title('Base image')
    axes[1].imshow(artificial_img.permute(1,2,0)/3000)
    axes[1].set_title('Artificial image')
    axes[2].imshow(delta_HR)
    axes[2].set_title('Delta')
    fig.suptitle('High resolution')
    comp = ImgComparator(fig, axes)
    plt.show()

    fig, axes = plt.subplots(1,3)

    axes[0].imshow(base_LR.permute(1,2,0)/3000)
    axes[0].set_title('Base image')
    axes[1].imshow(artificial_LR.permute(1,2,0)/3000)
    axes[1].set_title('Artificial image')
    axes[2].imshow(delta_LR)
    axes[2].set_title('Delta')
    fig.suptitle('Low resolution')
    comp = ImgComparator(fig, axes)
    plt.show()


    save_forpred = input('Do you want to save that pair for prediction later ? (y/n)')
    if 'y' in save_forpred:

        #Make folder to store the pair
        n_pairs = len(os.listdir(folder_drawing_pairs))
        os.mkdir(f'{folder_drawing_pairs}/{n_pairs}')

        

        # Get LR of artificial :
        #   First find the correct LR_data in the subds
        subdataset_base = base_img_id[0]
        idx_lr_data = find_lr_data_idx(base_LR, subdataset_base)

        lr_data_path = f'/localhome/iaga_dv/Dokumente/sat_data/cross_processed/{subdataset_base}/{idx_lr_data}/lr_data.tif'
        hr_data_path = f'/localhome/iaga_dv/Dokumente/sat_data/cross_processed/{subdataset_base}/{idx_lr_data}/hr_data.tif'

        with rasterio.open(lr_data_path) as src:
            lr_data = torch.from_numpy(src.read())


        #   Build the 3 first bands of DS(artificial) along with the band 7 of LR_data
        lr_data_7 = lr_data[7]
        artificial_LR_save = torch.cat([artificial_LR, lr_data_7.unsqueeze(0)], dim=0)
        base_LR_save = torch.cat([base_LR, lr_data_7.unsqueeze(0)], dim=0)


        # Save base_LR        
        save_into_tiff(base_LR_save, f'{folder_drawing_pairs}/{n_pairs}/base_LR.tif')

        #   Save the built LR artificial
        save_into_tiff(artificial_LR_save, f'{folder_drawing_pairs}/{n_pairs}/artificial_LR.tif')
        #Save the base_HR
        save_into_tiff(base_image, f'{folder_drawing_pairs}/{n_pairs}/base_HR.tif')
        #Save the artificial img (HR)
        save_into_tiff(artificial_img, f'{folder_drawing_pairs}/{n_pairs}/artificial_HR.tif')