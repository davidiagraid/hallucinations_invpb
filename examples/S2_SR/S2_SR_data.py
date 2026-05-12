import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import label, binary_dilation, binary_closing, gaussian_filter
from scipy.sparse import csr_matrix
from utils import apply_square_op_full, repatchify

def S2SR_collate(batch, S2_dataset, suffix):
    """
    Custom collate function for S2 super-resolution datasets.

    Args:
        batch: List of tuples containing (subdataset, image index, position).
        S2_dataset: Dataset object containing patch size and data path.
        suffix: Suffix for image files ('hr', 'lr', 'DSHR4', etc.).

    Returns:
        Stacked tensor of image patches.
    """
    if 'hr' in suffix:
        patchsize = S2_dataset.patchsizeX
    elif 'lr' in suffix or 'DSHR' in suffix:
        patchsize = S2_dataset.patchsizeY

    subdatasets, imgs_idxs, positions = zip(*batch)
    n = len(subdatasets)
    images_to_open = [(subdatasets[i], imgs_idxs[i]) for i in range(n)]
    images_to_open = list(set(images_to_open))
    opened_images = {}
    for (subdataset, img_idx) in images_to_open:
        if subdataset is not None:
            with rasterio.open(os.path.join(S2_dataset.data_path, subdataset, img_idx, f'{suffix}.tif')) as src:
                opened_images[(subdataset, img_idx)] = torch.from_numpy(src.read())

    n_bands = 3

    patches = []
    for k in range(n):
        if subdatasets[k] is not None:
            i, j = positions[k]
            patch = opened_images[(subdatasets[k], imgs_idxs[k])][:, i*patchsize:(i+1)*patchsize, j*patchsize:(j+1)*patchsize]
            patches.append(patch)
        else:
            patches.append(torch.zeros((n_bands, patchsize, patchsize)))

    return torch.stack(patches)

class S2_Dataloader(DataLoader):
    """
    Custom DataLoader for S2 super-resolution datasets using a custom collate function.
    """
    def __init__(self, dataset, suffix, **kwargs):
        self.dataset = dataset
        self.suffix = suffix

        def collate_fn(batch):
            return S2SR_collate(batch, self.dataset, self.suffix)

        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

class SRDataset_perimg_lightload(Dataset):
    """
    Dataset for loading super-resolution image patches per image without needing to store patches in separate files.

    Args:
        data_path: Path to the dataset.
        patchsizes: List of patch sizes (LR patchsize, HR patchsize).
        feasible_information_patches: Feasible information for patches.
        feasible_appartenance_patches: Feasible appartenance for patches.
        data_augmentation_type: Type of data augmentation.
        suffixes: Suffixes for image files.
        patch_suffixes: Suffixes for patch files.
        patched_shapes: Shapes of patched images per subdataset.
        P_null_path: Path to null operator numpy file.
    """
    def __init__(
        self,
        data_path,
        patchsizes,
        feasible_information_patches,
        feasible_appartenance_patches,
        data_augmentation_type='None',
        suffixes=('lr', 'hr'),
        patch_suffixes=('lr', 'hr'),
        patched_shapes={'naip': (40, 40), 'spain_crops': (42, 42), 'spain_urban': (42, 42), 'spot': (42, 42)},
        P_null_path='/localhome/iaga_dv/Dokumente/Operators/P_null_32.npy'
    ):
        self.subdatasets = [x for x in patched_shapes.keys()]

        self.data_path = data_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]
        self.patched_shape = patched_shapes
        self.suffix_list_patches = patch_suffixes if not isinstance(patch_suffixes, str) else [patch_suffixes]
        if 'lr' in self.suffix_list_patches[0] or 'DSHR' in self.suffix_list_patches[0]:
            self.lr_suffix = self.suffix_list_patches[0]
            self.hr_suffix = self.suffix_list_patches[1]
        elif 'hr' in self.suffix_list_patches[0]:
            self.lr_suffix = self.suffix_list_patches[1]
            self.hr_suffix = self.suffix_list_patches[0]

        def verify_path(filepath):
            for subds in self.subdatasets:
                if subds in filepath:
                    return True
            return False

        self.img_file_list = sorted(
            [f for f in glob.glob(os.path.join(data_path, '**', '*'), recursive=True)
             if (self.suffix_list[0] in os.path.basename(f) and f.endswith('.tif') and verify_path(f))]
        )

        self.patch_ids = []
        self.image_ids = []
        imgs_idx_list = {}

        subdatasets = ['naip', 'spain_crops', 'spain_urban', 'spot']

        current_idx = 0
        for img_file in self.img_file_list:
            for subds in subdatasets:
                if subds in img_file:
                    subds_img = subds

            patched_shape = self.patched_shape[subds_img]
            nb_patches_img = patched_shape[0] * patched_shape[1]

            nb_img = os.path.normpath(img_file).split(os.sep)[-2]
            subds_img = os.path.normpath(img_file).split(os.sep)[-3]

            self.image_ids.append((subds_img, nb_img))
            imgs_idx_list[(subds_img, nb_img)] = list(range(current_idx, current_idx + nb_patches_img))

            for i in range(nb_patches_img):
                self.patch_ids.append((subds_img, nb_img, i))
            current_idx += nb_patches_img

        self.imgs_idx_list = imgs_idx_list

        if data_augmentation_type not in ['None', 'standard', 'symetric', 'standardsym']:
            raise Exception(f"Incorrect data augmentation type. You gave {data_augmentation_type} Supported ones: 'None','standard', 'symetric', 'standardsym'")
        self.data_augm_type = data_augmentation_type
        len_factors = {'None': 1, 'standard': 3, 'symetric': 2, 'standardsym': 6}
        self.len_factor = len_factors[data_augmentation_type]

        if len(patchsizes) == 2:
            patchsize_0, patchsize_1 = patchsizes
            self.patchsizeX = max(patchsize_0, patchsize_1)
            self.patchsizeY = min(patchsize_0, patchsize_1)
            self.SR_factor = self.patchsizeX // self.patchsizeY
        elif len(patchsizes) == 1:
            if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                self.patchsizeY = patchsizes[0]
                self.patchsizeX = patchsizes[0] * 4
            elif 'hr' in self.suffix_list[0]:
                self.patchsizeX = patchsizes[0]
                self.patchsizeY = patchsizes[0] // 4
            else:
                print('lr or hr not in the suffix names. Please add one of them.')
                print(f'Your suffixes are {suffixes}')

            self.SR_factor = self.patchsizeX // self.patchsizeY
        else:
            print('Only 1 or 2 suffixes are allowed')
            print(f'You gave suffixes = {suffixes}')

        self.lim_area_ratio = 0.9
        self.n_iter_max_ratio = 1.0
        self.sigma_blend = 1

        self.border = 16
        self.P_null = np.load(P_null_path)

        self.feas_info_patches = feasible_information_patches
        self.feas_app_patches = feasible_appartenance_patches

    def __len__(self):
        if len(self.suffix_list) == 1:
            if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                return len(self.image_ids)
            elif 'hr' in self.suffix_list[0]:
                return self.len_factor * len(self.image_ids)
        elif len(self.suffix_list) == 2:
            return self.len_factor * len(self.image_ids)
        else:
            raise Exception(f"Invalid set of suffixes. You gave {self.suffix_list}")

    def __getitem__(self, idx):
        idx_base = idx // self.len_factor
        image_id = self.image_ids[idx_base]
        subdataset, idx_in_subds = image_id

        lr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.lr_suffix)

        if self.data_augm_type == 'None':
            if len(self.suffix_list) == 1:
                if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                    return lr_image
                elif 'hr' in self.suffix_list[0]:
                    hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                    return hr_image
            elif len(self.suffix_list) == 2:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict

        elif self.data_augm_type == 'standard':
            if len(self.suffix_list) == 1 and ('lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]):
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list) == 2 and idx % self.len_factor == 0:
                res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_img')}"}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict
            elif len(self.suffix_list) == 1 and idx % self.len_factor == 0 and 'hr' in self.suffix_list[0]:
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                return hr_image

            Fy_lim1, Fy_lim2, _, _ = self.get_Fy_fullimg_V2(
                subdataset, str(idx_in_subds), self.feas_info_patches, self.feas_app_patches,
                self.hr_suffix, self.lr_suffix, lim_area_ratio=self.lim_area_ratio,
                n_iter_max_ratio=self.n_iter_max_ratio, sigma_blend=self.sigma_blend, margin_blend=1
            )

            if len(self.suffix_list) == 1 and 'hr' in self.suffix_list[0]:
                if idx % self.len_factor == 1:
                    return Fy_lim1
                elif idx % self.len_factor == 2:
                    return Fy_lim2
                else:
                    raise Exception("Unexpected error in data augmentation.")
            elif len(self.suffix_list) == 2:
                if idx % self.len_factor == 1:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim1')}"}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 2:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim2')}"}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                return res_dict

        elif self.data_augm_type == 'standardsym':
            hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)

            if len(self.suffix_list) == 1 and ('lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]):
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list) == 2 and idx % self.len_factor == 0:
                res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_img')}"}
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict
            elif len(self.suffix_list) == 1 and idx % self.len_factor == 0 and 'hr' in self.suffix_list[0]:
                return hr_image

            Fy_lim1, Fy_lim2, _, _ = self.get_Fy_fullimg_V2(
                subdataset, str(idx_in_subds), self.feas_info_patches, self.feas_app_patches,
                self.hr_suffix, self.lr_suffix, lim_area_ratio=self.lim_area_ratio,
                n_iter_max_ratio=self.n_iter_max_ratio, sigma_blend=self.sigma_blend, margin_blend=1
            )

            hr_null = apply_square_op_full(self.P_null, hr_image, out_2D_shape_op=(128, 128), border=self.border)
            lim1_null = apply_square_op_full(self.P_null, Fy_lim1, out_2D_shape_op=(128, 128), border=self.border)
            lim2_null = apply_square_op_full(self.P_null, Fy_lim2, out_2D_shape_op=(128, 128), border=self.border)

            hr_sym = hr_image - 2 * hr_null
            lim1_sym = Fy_lim1 - 2 * lim1_null
            lim2_sym = Fy_lim2 - 2 * lim2_null

            if len(self.suffix_list) == 1 and 'hr' in self.suffix_list[0]:
                if idx % self.len_factor == 1:
                    return Fy_lim1
                elif idx % self.len_factor == 2:
                    return Fy_lim2
                elif idx % self.len_factor == 3:
                    return hr_sym
                elif idx % self.len_factor == 4:
                    return lim1_sym
                elif idx % self.len_factor == 5:
                    return lim2_sym
                else:
                    raise Exception("Unexpected error in data augmentation.")
            elif len(self.suffix_list) == 2:
                if idx % self.len_factor == 1:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim1')}"}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 2:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim2')}"}
                    res_dict[self.hr_suffix] = Fy_lim2
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 3:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_sym')}"}
                    res_dict[self.hr_suffix] = hr_sym
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 4:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim1_sym')}"}
                    res_dict[self.hr_suffix] = lim1_sym
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 5:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim2_sym')}"}
                    res_dict[self.hr_suffix] = lim2_sym
                    res_dict[self.lr_suffix] = lr_image
                else:
                    raise Exception("Unexpected error in data augmentation.")
                return res_dict

        elif self.data_augm_type == 'symetric':
            idx_base = idx // self.len_factor
            image_id = self.image_ids[idx_base]
            subdataset, idx_in_subds = image_id

            lr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.lr_suffix)
            hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)

            hr_null = apply_square_op_full(self.P_null, hr_image, out_2D_shape_op=(128, 128), border=self.border)
            hr_sym = hr_image - 2 * hr_null

            if len(self.suffix_list) == 1 and ('lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]):
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list) == 1 and 'hr' in self.suffix_list[0]:
                if idx % self.len_factor == 0:
                    return hr_image
                elif idx % self.len_factor == 1:
                    return hr_sym
                else:
                    raise Exception("Unexpected error in data augmentation.")
            elif len(self.suffix_list) == 2:
                if idx % self.len_factor == 0:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_img')}"}
                    res_dict[self.lr_suffix] = lr_image
                    res_dict[self.hr_suffix] = hr_image
                elif idx % self.len_factor == 1:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_sym')}"}
                    res_dict[self.hr_suffix] = hr_sym
                else:
                    raise Exception("Unexpected error in data augmentation.")
                return res_dict

    def get_full_img(self, subdataset, img_idx, suffix):
        """
        Loads a full image from disk and crops it to patch size.

        Args:
            subdataset: Subdataset name.
            img_idx: Image index.
            suffix: Suffix for image file.

        Returns:
            Cropped image tensor.
        """
        img_path = os.path.join(self.data_path, subdataset, img_idx, f'{suffix}.tif')
        if 'lr' in suffix or 'DSHR' in suffix:
            PS = self.patchsizeY
        elif 'hr' in suffix:
            PS = self.patchsizeX

        with rasterio.open(img_path) as src:
            img = torch.from_numpy(src.read())

        c, h, w = img.shape
        return img[:, :h - h % PS, :w - w % PS]

    def get_patch_position(self, idx_in_img, patched_shape):
        """
        Converts a 1D patch index to 2D position.

        Args:
            idx_in_img: Patch index in image.
            patched_shape: Shape of patched image.

        Returns:
            (row, col) position.
        """
        nr, nc = patched_shape
        row = idx_in_img // nc
        col = idx_in_img % nc
        return row, col

    def get_search_shift(self, position_base, position_lim, patched_shape_base, patched_shape_lim):
        """
        Computes search shift between base and limit positions.

        Args:
            position_base: Base position.
            position_lim: Limit position.
            patched_shape_base: Shape of base image.
            patched_shape_lim: Shape of limit image.

        Returns:
            (y_min, y_max, x_min, x_max) search shifts.
        """
        i_base, j_base = position_base
        i_lim, j_lim = position_lim
        nr_base, nc_base = patched_shape_base
        nr_lim, nc_lim = patched_shape_lim

        y_min, y_max = -min(i_base, i_lim), min(nr_base - i_base, nr_lim - i_lim)
        x_min, x_max = -min(j_base, j_lim), min(nc_base - j_base, nc_lim - j_lim)
        return y_min, y_max, x_min, x_max

    def get_same_feasible_area(self, position_base, position_lim, search_shift_lim, feas_app, base_img_id, lim_img_id, patched_shape_lim, patched_shape_base):
        """
        Finds the maximum connected area in the limit image corresponding to the base image patch
        such that every patch in the lim image is in a same feasible set as the corresponding patch in the base image.

        Args:
            position_base: Position in base image.
            position_lim: Position in limit image.
            search_shift_lim: Search shift.
            feas_app: Feasible appartenance matrix.
            base_img_id: Base image ID.
            lim_img_id: Limit image ID.
            patched_shape_lim: Shape of limit image.
            patched_shape_base: Shape of base image.

        Returns:
            (mask_same_feasible_lim, replacement_idx)
        """
        i_lim, j_lim = position_lim
        i_base, j_base = position_base

        mask_same_feasible_lim = np.zeros(patched_shape_lim)
        replacement_idx = np.zeros(patched_shape_lim)
        sh_ymin, sh_ymax, sh_xmin, sh_xmax = search_shift_lim

        for i in range(sh_ymin, sh_ymax):
            for j in range(sh_xmin, sh_xmax):
                pos_1D_base = self.get_patch_idx_img((i + i_base, j + j_base), patched_shape_base)
                idx_base = self.imgs_idx_list[base_img_id][pos_1D_base]

                pos1D_lim = self.get_patch_idx_img((i + i_lim, j + j_lim), patched_shape_lim)
                idx_lim = self.imgs_idx_list[lim_img_id][pos1D_lim]

                mask_same_feasible_lim[i + i_lim, j + j_lim] = feas_app[idx_lim, idx_base]
                if mask_same_feasible_lim[i + i_lim, j + j_lim] > 0.5:
                    replacement_idx[i + i_lim, j + j_lim] = idx_lim

                if i == 0 and j == 0 and feas_app[idx_lim, idx_base] == 0 and True:
                    patch_id_base = self.patch_ids[idx_base]
                    img_base_id = self.get_img_id(int(idx_base))
                    subds_img_base = img_base_id[0]
                    position_1D_base = int(patch_id_base.split('_')[-1])
                    position_2D_base = self.get_patch_position(position_1D_base, self.patched_shape[subds_img_base])
                    print('Unexpected behaviour in the image reconstitution')
                    print(f'Looked Base position: {position_2D_base}')
                    print(f'Looked indexes: {idx_lim}, {idx_base}')

        labeled_array, num_features = label(mask_same_feasible_lim)
        component_label = labeled_array[i_lim, j_lim]

        if component_label == 0:
            return np.zeros_like(mask_same_feasible_lim, dtype=bool), np.zeros_like(mask_same_feasible_lim)

        mask_same_feasible_lim = labeled_array == component_label
        replacement_idx = replacement_idx * mask_same_feasible_lim
        return mask_same_feasible_lim, replacement_idx

    def get_same_feasible_area_nobaseX(self, position_base, position_lim, search_shift_lim, feas_app_new, lim_img_id, patched_shape_lim, patched_shape_base):
        """
        Finds the maximum connected area in the limit image corresponding to the base image patch
        such that every patch in the lim image is in a same feasible set as the corresponding patch in the base image.

        Args:
            position_base: Position in base image.
            position_lim: Position in limit image.
            search_shift_lim: Search shift.
            feas_app_new: Feasible appartenance matrix.
            lim_img_id: Limit image ID.
            patched_shape_lim: Shape of limit image.
            patched_shape_base: Shape of base image.

        Returns:
            (mask_same_feasible_lim, replacement_idx)
        """
        i_lim, j_lim = position_lim
        i_base, j_base = position_base

        mask_same_feasible_lim = np.zeros(patched_shape_lim)
        replacement_idx = np.zeros(patched_shape_lim)
        sh_ymin, sh_ymax, sh_xmin, sh_xmax = search_shift_lim

        for i in range(sh_ymin, sh_ymax):
            for j in range(sh_xmin, sh_xmax):
                pos_1D_base = self.get_patch_idx_img((i + i_base, j + j_base), patched_shape_base)
                idx_base = pos_1D_base

                pos1D_lim = self.get_patch_idx_img((i + i_lim, j + j_lim), patched_shape_lim)
                idx_lim = self.imgs_idx_list[lim_img_id][pos1D_lim]

                mask_same_feasible_lim[i + i_lim, j + j_lim] = feas_app_new[idx_lim, idx_base]
                if mask_same_feasible_lim[i + i_lim, j + j_lim] > 0.5:
                    replacement_idx[i + i_lim, j + j_lim] = idx_lim

        labeled_array, num_features = label(mask_same_feasible_lim)
        component_label = labeled_array[i_lim, j_lim]

        if component_label == 0:
            return np.zeros_like(mask_same_feasible_lim, dtype=bool), np.zeros_like(mask_same_feasible_lim)

        mask_same_feasible_lim = labeled_array == component_label
        replacement_idx = replacement_idx * mask_same_feasible_lim
        return mask_same_feasible_lim, replacement_idx

    def shift_mask(self, mask, k, l, patched_shape_dst):
        """
        Shifts a mask by (k, l) and fits it to the destination shape.

        Args:
            mask: Mask to shift.
            k: Row shift.
            l: Column shift.
            patched_shape_dst: Destination shape.

        Returns:
            Shifted mask.
        """
        shifted_res = np.zeros_like(mask)

        rows, cols = mask.shape
        nr_dst, nc_dst = patched_shape_dst

        src_row_start = max(0, -k)
        src_row_end = rows - max(0, k)
        src_col_start = max(0, -l)
        src_col_end = cols - max(0, l)

        dst_row_start = max(0, k)
        dst_row_end = rows - max(0, -k)
        dst_col_start = max(0, l)
        dst_col_end = cols - max(0, -l)

        shifted_res[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mask[src_row_start:src_row_end, src_col_start:src_col_end]
        if rows <= nr_dst:
            shifted = np.zeros(patched_shape_dst)
            shifted[:rows, :cols] = shifted_res
        else:
            shifted = shifted_res[:nr_dst, nc_dst]
        return shifted

    def get_patch_idx_img(self, position, patched_shape):
        """
        Converts a 2D patch position to 1D index.

        Args:
            position: (row, col) position.
            patched_shape: Shape of patched image.

        Returns:
            1D patch index.
        """
        i, j = position
        nr, nc = patched_shape
        return i * nc + j

    def fill_patch_info(self, values_list, patched_shape):
        """
        Fills a 2D grid with values from a list according to patch positions.

        Args:
            values_list: List of values for patches.
            patched_shape: Shape of patched image.

        Returns:
            2D tensor of patch info.
        """
        nr, nc = patched_shape
        patch_info = torch.zeros((nr, nc))
        for idx in range(len(values_list)):
            i, j = self.get_patch_position(idx, patched_shape=patched_shape)
            patch_info[int(i), int(j)] = values_list[int(idx)]
        return patch_info

    def get_2D_patchgrid(self, idx_list, suffix, patched_shape, default_image=torch.zeros((3, 5000, 5000))):
        """
        Loads patches and arranges them in a 2D grid.

        Args:
            idx_list: List of patch indices.
            suffix: Suffix for patch files.
            patched_shape: Shape of patched image.
            default_image: Default image if patch is missing.

        Returns:
            5D tensor of patches.
        """
        assert suffix in self.suffix_list_patches

        if 'lr' in suffix or 'DSHR' in suffix:
            PS = self.patchsizeY
        elif 'hr' in suffix:
            PS = self.patchsizeX
        c, h, w = default_image.shape
        nr, nc = h // PS, w // PS

        patches_id = []
        for idx in idx_list:
            if idx == -1:
                patches_id.append((None, None, None))
            else:
                patches_id.append(self.patch_ids[idx])

        imgs_ids = [(subds, nb_img) for (subds, nb_img, _) in patches_id]
        imgs_ids = list(set(imgs_ids))

        imgs = {}
        for img_id in imgs_ids:
            if img_id[0] is not None:
                path = os.path.join(self.data_path, img_id[0], img_id[1], f'{suffix}.tif')
                with rasterio.open(path) as src:
                    imgs[img_id] = torch.from_numpy(src.read())
        patches = []
        count = 0
        for (subds, nb_img, pos_1D) in patches_id:
            if subds is not None:
                img = imgs[(subds, nb_img)]
                i, j = self.get_patch_position(pos_1D, patched_shape=patched_shape)
                patch = img[:, i * PS:(i + 1) * PS, j * PS: (j + 1) * PS]
                if patch.shape != (c, PS, PS):
                    patch = default_image[:, i * PS:(i + 1) * PS, j * PS: (j + 1) * PS]
            else:
                i, j = self.get_patch_position(count, (nr, nc))
                patch = default_image[:, i * PS:(i + 1) * PS, j * PS: (j + 1) * PS]
            count += 1
            patches.append(patch)

        patches = torch.stack(patches)
        c = patches.shape[1]
        ps = patches.shape[2]

        n_patches_y, n_patches_x = patched_shape
        patches = patches.view(n_patches_y, n_patches_x, c, ps, ps)

        return patches

    def recompose_image(self, idx_list, suffix, patched_shape, default_image=torch.zeros((3, 5000, 5000))):
        """
        Recomposes an image from its patches.

        Args:
            idx_list: List of patch indices.
            suffix: Suffix for patch files.
            patched_shape: Shape of patched image.
            default_image: Default image if patch is missing.

        Returns:
            Recomposed image tensor.
        """
        patches = self.get_2D_patchgrid(idx_list, suffix, patched_shape, default_image)

        n_patches_y, n_patches_x, c, ps, ps = patches.shape
        reconstructed = patches.permute(2, 0, 3, 1, 4)
        reconstructed = reconstructed.reshape(c, n_patches_y * ps, n_patches_x * ps)

        return reconstructed

    def amplify_mask(self, small_mask, size, patched_shape):
        """
        Amplifies a small mask to the size of the full image.

        Args:
            small_mask: Small mask.
            size: 'big' or 'small'.
            patched_shape: Shape of patched image.

        Returns:
            Amplified mask.
        """
        nr, nc = patched_shape
        if size == 'big':
            ps = self.patchsizeX
        elif size == 'small':
            ps = self.patchsizeY
        else:
            print('Please choose a size among big or small')
            raise ValueError
        big_mask = np.zeros((nr * ps, nc * ps))

        for i in range(nr):
            for j in range(nc):
                big_mask[i * ps: (i + 1) * ps, j * ps: (j + 1) * ps] = small_mask[i, j]
        return big_mask

    def get_Fy_fullimg_idx_V2(self, subdataset, img_idx_insubds, feasible_info, feas_app, lim_area_ratio=0.6, n_iter_max_ratio=0.4):
        """
        Computes patch replacement indices and orders for limit images.

        Args:
            subdataset: Subdataset name.
            img_idx_insubds: Image index in subdataset.
            feasible_info: Feasible information.
            feas_app: Feasible appartenance matrix.
            lim_area_ratio: Area ratio for limit.
            n_iter_max_ratio: Max iteration ratio.

        Returns:
            Replacement indices and orders for limit images.
        """
        img_idx_list = self.imgs_idx_list[(subdataset, img_idx_insubds)]

        Fy_infos = [feasible_info[idx] for idx in img_idx_list]

        diams_Fy = torch.tensor([info[0] for info in Fy_infos])
        Fy_lim1_idx = torch.tensor([info[1][0] for info in Fy_infos])
        Fy_lim2_idx = torch.tensor([info[1][1] for info in Fy_infos])

        Fy_lim1_idx_small = self.fill_patch_info(Fy_lim1_idx, self.patched_shape[subdataset])
        Fy_lim2_idx_small = self.fill_patch_info(Fy_lim2_idx, self.patched_shape[subdataset])

        diams_Fy_small = self.fill_patch_info(diams_Fy, self.patched_shape[subdataset])

        patched_shape = self.patched_shape[subdataset]
        flat_diams = diams_Fy_small.flatten()
        sorted_values, sorted_indices = torch.sort(flat_diams, descending=True)
        nc = patched_shape[1]
        rows = sorted_indices // nc
        cols = sorted_indices % nc
        indices_2d_diams = torch.stack((rows, cols), dim=1)

        replace_lim1_idx = np.zeros(patched_shape)
        replace_lim2_idx = np.zeros(patched_shape)
        replace_lim1_order = np.zeros(patched_shape)
        replace_lim2_order = np.zeros(patched_shape)

        replaced_prop_lim1 = 0
        replaced_prop_lim2 = 0

        n_iter = 1
        n_replace = 1

        nr, nc = patched_shape
        structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        while n_iter < nr * nc * n_iter_max_ratio and replaced_prop_lim1 < lim_area_ratio and replaced_prop_lim2 < lim_area_ratio:
            maxdiam_position = indices_2d_diams[n_iter, :]
            maxdiam_position = int(maxdiam_position[0]), int(maxdiam_position[1])
            i_base, j_base = maxdiam_position

            idx_Fy_lim1 = Fy_lim1_idx_small[i_base, j_base]
            idx_Fy_lim2 = Fy_lim2_idx_small[i_base, j_base]

            lim_patch1_id = self.patch_ids[int(idx_Fy_lim1)]
            lim_patch2_id = self.patch_ids[int(idx_Fy_lim2)]

            patched_shape_base = self.patched_shape[subdataset]

            img_lim1_id = self.get_img_id(int(idx_Fy_lim1))
            subds_img_lim1 = img_lim1_id[0]
            patched_shape_lim1 = self.patched_shape[subds_img_lim1]
            patch_lim1_nb = lim_patch1_id.split('_')[-1]
            position_img_lim1 = self.get_patch_position(int(patch_lim1_nb), patched_shape=patched_shape_lim1)
            i_lim1, j_lim1 = position_img_lim1
            search_shift_lim1 = self.get_search_shift(maxdiam_position, position_img_lim1, patched_shape_base, patched_shape_lim1)
            same_feas_area_lim1, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim1, search_shift_lim1, feas_app, (subdataset, img_idx_insubds), img_lim1_id, patched_shape_lim1, patched_shape_base)
            same_feas_area_lim1_shifted, replacement_idx_shifted_lim1 = self.shift_mask(same_feas_area_lim1, i_base - i_lim1, j_base - j_lim1, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base - i_lim1, j_base - j_lim1, patched_shape_dst=patched_shape_base)

            img_lim2_id = self.get_img_id(int(idx_Fy_lim2))
            subds_img_lim2 = img_lim2_id[0]
            patched_shape_lim2 = self.patched_shape[subds_img_lim2]
            patch_lim2_nb = lim_patch2_id.split('_')[-1]
            position_img_lim2 = self.get_patch_position(int(patch_lim2_nb), patched_shape=patched_shape_lim2)
            i_lim2, j_lim2 = position_img_lim2
            search_shift_lim2 = self.get_search_shift(maxdiam_position, position_img_lim2, patched_shape_base, patched_shape_lim2)
            same_feas_area_lim2, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim2, search_shift_lim2, feas_app, (subdataset, img_idx_insubds), img_lim2_id, patched_shape_lim2, patched_shape_base)
            same_feas_area_lim2_shifted, replacement_idx_shifted_lim2 = self.shift_mask(same_feas_area_lim2, i_base - i_lim2, j_base - j_lim2, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base - i_lim2, j_base - j_lim2, patched_shape_dst=patched_shape_base)

            iterative_mask_lim1 = replace_lim1_idx > 0.5
            iterative_mask_lim2 = replace_lim2_idx > 0.5

            dilated_iterative_mask_lim1 = binary_dilation(iterative_mask_lim1, structure=structuring_element)
            dilated_iterative_mask_lim2 = binary_dilation(iterative_mask_lim2, structure=structuring_element)

            intersection_lim1 = np.logical_and(same_feas_area_lim1_shifted, dilated_iterative_mask_lim1)
            intersection_lim2 = np.logical_and(same_feas_area_lim2_shifted, dilated_iterative_mask_lim2)

            if not np.any(intersection_lim1) and not np.any(intersection_lim2):
                replace_lim1_idx = np.where(same_feas_area_lim1_shifted > 0.5, replacement_idx_shifted_lim1, replace_lim1_idx)
                replace_lim2_idx = np.where(same_feas_area_lim2_shifted > 0.5, replacement_idx_shifted_lim2, replace_lim2_idx)
                replace_lim1_order = np.where(same_feas_area_lim1_shifted > 0.5, n_replace, replace_lim1_order)
                replace_lim2_order = np.where(same_feas_area_lim2_shifted > 0.5, n_replace, replace_lim2_order)
                n_replace += 1

            n_iter += 1

            closed_mask_lim1 = binary_closing(replace_lim1_order > 0.5, structure=structuring_element)
            closed_mask_lim2 = binary_closing(replace_lim2_order > 0.5, structure=structuring_element)

            replaced_prop_lim1 = np.sum(closed_mask_lim1 > 0.5) / (patched_shape[0] * patched_shape[1])
            replaced_prop_lim2 = np.sum(closed_mask_lim2 > 0.5) / (patched_shape[0] * patched_shape[1])

        return replace_lim1_idx, replace_lim2_idx, replace_lim1_order, replace_lim2_order

    def get_Fy_fullimg_V2(self, subdataset, img_idx_insubds, feasible_info, feas_app, hr_suffix='hr_res', lr_suffix='lr_res', lim_area_ratio=0.9, n_iter_max_ratio=1.0, sigma_blend=1, margin_blend=1):
        """
        Generates limit images from the outputs of get_Fy_fullimg_idx_V2 using patch blending for pasting.

        Args:
            subdataset: Subdataset name.
            img_idx_insubds: Image index in subdataset.
            feasible_info: Feasible information.
            feas_app: Feasible appartenance matrix.
            hr_suffix: Suffix for high-resolution images.
            lr_suffix: Suffix for low-resolution images.
            lim_area_ratio: Area ratio for limit.
            n_iter_max_ratio: Max iteration ratio.
            sigma_blend: Sigma for Gaussian blending.
            margin_blend: Margin for blending.

        Returns:
            Limit images and their low-resolution versions.
        """
        replace_lim1_idx, replace_lim2_idx, replace_lim1_order, replace_lim2_order = self.get_Fy_fullimg_idx_V2(
            subdataset=subdataset, img_idx_insubds=img_idx_insubds, feasible_info=feasible_info, feas_app=feas_app,
            lim_area_ratio=lim_area_ratio, n_iter_max_ratio=n_iter_max_ratio
        )
        replace_lim1_idx_flat, replace_lim2_idx_flat, replace_lim1_order_flat, replace_lim2_order_flat = replace_lim1_idx.flatten(), replace_lim2_idx.flatten(), replace_lim1_order.flatten(), replace_lim2_order.flatten()
        baseimg_idx_list = np.array(self.imgs_idx_list[(subdataset, img_idx_insubds)])

        blending = True
        if not blending:
            img_lim1_idx_flat = np.where(replace_lim1_order_flat > 0.5, replace_lim1_idx_flat, baseimg_idx_list).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat > 0.5, replace_lim2_idx_flat, baseimg_idx_list).astype(int)

            Fy_lim1_img = self.recompose_image(img_lim1_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_img = self.recompose_image(img_lim2_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])
        else:
            base_imgX = self.recompose_image(baseimg_idx_list, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            img_lim1_idx_flat = np.where(replace_lim1_order_flat > 0.5, replace_lim1_idx_flat, baseimg_idx_list).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat > 0.5, replace_lim2_idx_flat, baseimg_idx_list).astype(int)

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])

            res_mask_lim1 = replace_lim1_order > 0.5
            res_bigmask_lim1 = self.amplify_mask(res_mask_lim1, size='big', patched_shape=self.patched_shape[subdataset])

            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(res_bigmask_lim1, kernel, iterations=1)
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma=sigma_blend))

            Fy_lim1_img_noblend = self.recompose_image(img_lim1_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim1_img = blurred_mask * Fy_lim1_img_noblend + (1 - blurred_mask) * base_imgX

            res_mask_lim2 = replace_lim2_order > 0.5
            res_bigmask_lim2 = self.amplify_mask(res_mask_lim2, size='big', patched_shape=self.patched_shape[subdataset])

            eroded_mask = cv2.erode(res_bigmask_lim2, kernel, iterations=1)
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma=sigma_blend))

            Fy_lim2_img_noblend = self.recompose_image(img_lim2_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_img = blurred_mask * Fy_lim2_img_noblend + (1 - blurred_mask) * base_imgX

        return Fy_lim1_img, Fy_lim2_img, Fy_lim1_imgY, Fy_lim2_imgY

class SRDataset_lightload(Dataset):
    """
    Lightweight dataset for loading super-resolution image patches.
    """
    def __init__(
        self,
        data_path,
        patchsizes,
        suffixes=('lr', 'hr'),
        patched_shapes={'naip': (40, 40), 'spain_crops': (42, 42), 'spain_urban': (42, 42), 'spot': (42, 42)}
    ):
        self.subdatasets = [x for x in patched_shapes.keys()]

        self.data_path = data_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]
        self.patched_shape = patched_shapes

        def verify_path(filepath):
            for subds in self.subdatasets:
                if subds in filepath:
                    return True
            return False

        self.img_file_list = sorted(
            [f for f in glob.glob(os.path.join(data_path, '**', '*'), recursive=True)
             if (self.suffix_list[0] in os.path.basename(f) and f.endswith('.tif') and verify_path(f))]
        )

        self.patch_ids = []
        self.image_ids = []
        imgs_idx_list = {}

        subdatasets = ['naip', 'spain_crops', 'spain_urban', 'spot']

        current_idx = 0
        for img_file in self.img_file_list:
            for subds in subdatasets:
                if subds in img_file:
                    subds_img = subds

            patched_shape = self.patched_shape[subds_img]
            nb_patches_img = patched_shape[0] * patched_shape[1]
            nb_img = os.path.normpath(img_file).split(os.sep)[-2]
            subds_img = os.path.normpath(img_file).split(os.sep)[-3]

            self.image_ids.append((subds_img, nb_img))
            imgs_idx_list[(subds_img, nb_img)] = list(range(current_idx, current_idx + nb_patches_img))

            for i in range(nb_patches_img):
                self.patch_ids.append((subds_img, nb_img, i))
            current_idx += nb_patches_img

        self.imgs_idx_list = imgs_idx_list

        if len(patchsizes) == 2:
            patchsize_0, patchsize_1 = patchsizes
            self.patchsizeX = max(patchsize_0, patchsize_1)
            self.patchsizeY = min(patchsize_0, patchsize_1)
            self.SR_factor = self.patchsizeX // self.patchsizeY
        elif len(patchsizes) == 1:
            if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                self.patchsizeY = patchsizes[0]
                self.patchsizeX = patchsizes[0] * 4
            elif 'hr' in self.suffix_list[0]:
                self.patchsizeX = patchsizes[0]
                self.patchsizeY = patchsizes[0] // 4
            else:
                print('lr or hr not in the suffix names. Please add one of them.')
                print(f'Your suffixes are {suffixes}')

            self.SR_factor = self.patchsizeX // self.patchsizeY
        else:
            print('Only 1 or 2 suffixes are allowed')
            print(f'You gave suffixes = {suffixes}')

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        subds, img_idx, idx_in_img = self.patch_ids[idx]
        i, j = self.get_patch_position(idx_in_img, patched_shape=self.patched_shape[subds])
        return (subds, img_idx, (i, j))

    def get_patch_position(self, idx_in_img, patched_shape):
        nr, nc = patched_shape
        row = idx_in_img // nc
        col = idx_in_img % nc
        return row, col

    def get_patch_idx_img(self, position, patched_shape):
        i, j = position
        nr, nc = patched_shape
        return i * nc + j

class SRDataset_perimg(Dataset):
    """
    Dataset for loading super-resolution image patches per image.
    """
    def __init__(
        self,
        folder_path,
        feasible_information_patches,
        feasible_appartenance_patches,
        suffixes=('lr', 'hr'),
        patch_suffixes=('lr', 'hr'),
        data_augmentation_type='None',
        patched_shapes={'naip': (40, 40), 'spain_crops': (42, 42), 'spain_urban': (42, 42), 'spot': (42, 42)},
        P_null_path='/localhome/iaga_dv/Dokumente/Operators/P_null_32.npy'
    ):
        def get_img_id(patch_id):
            splitted_patch_id = patch_id.split('_')
            idx_img = splitted_patch_id[-2]
            if len(splitted_patch_id) == 4:
                subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
            elif len(splitted_patch_id) == 3:
                subds = splitted_patch_id[0]
            return subds, idx_img

        self.folder_path = folder_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]
        self.suffix_list_patches = patch_suffixes if not isinstance(patch_suffixes, str) else [patch_suffixes]
        if 'lr' in self.suffix_list_patches[0] or 'DSHR' in self.suffix_list_patches[0]:
            self.lr_suffix = self.suffix_list_patches[0]
            self.hr_suffix = self.suffix_list_patches[1]
        elif 'hr' in self.suffix_list_patches[0]:
            self.lr_suffix = self.suffix_list_patches[1]
            self.hr_suffix = self.suffix_list_patches[0]
        else:
            raise Exception(f"Invalid patch suffixes. You gave {self.suffix_list_patches}")

        self.feas_info_patches = feasible_information_patches
        self.feas_app_patches = feasible_appartenance_patches

        self.file_list = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list_patches[0]}.tif')))
        self.patched_shape = patched_shapes

        self.patch_ids = [os.path.basename(f).replace(f'_{self.suffix_list_patches[0]}.tif', '') for f in self.file_list]

        self.image_ids = list(set([get_img_id(p_id) for p_id in self.patch_ids]))
        imgs_idx_list = {}
        for img_id in self.image_ids:
            subds, n_img = img_id

            img_idx_list = [(i, self.patch_ids[i].split('_')[-1]) for i in range(len(self.patch_ids)) if (f'{subds}_{n_img}_' in self.patch_ids[i])]
            img_idx_list = sorted(
                img_idx_list,
                key=lambda x: int(x[1])
            )
            img_idx_list = [int(x[0]) for x in img_idx_list]
            imgs_idx_list[img_id] = img_idx_list

        self.imgs_idx_list = imgs_idx_list
        self.patched_shape = patched_shapes

        if data_augmentation_type not in ['None', 'standard', 'symetric', 'standardsym']:
            raise Exception(f"Incorrect data augmentation type. You gave {data_augmentation_type} Supported ones: 'None','standard', 'symetric', 'standardsym'")
        self.data_augm_type = data_augmentation_type
        len_factors = {'None': 1, 'standard': 3, 'symetric': 2, 'standardsym': 6}
        self.len_factor = len_factors[data_augmentation_type]

        custom_feas_app = np.zeros((self.len_factor * len(self.image_ids), len(self.image_ids)))
        for y_idx in range(len(self.image_ids)):
            custom_feas_app[self.len_factor * y_idx: self.len_factor * (y_idx + 1), y_idx] = 1
        self.feas_app_fullimg = csr_matrix(custom_feas_app)

        if len(self.suffix_list_patches) == 2:
            self.file_list1 = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list_patches[1]}.tif')))

            with rasterio.open(self.file_list1[0]) as src:
                patch_1 = torch.from_numpy(src.read())
                patchsize_1 = patch_1.shape[-1]
            with rasterio.open(self.file_list[0]) as src:
                patch_0 = torch.from_numpy(src.read())
                patchsize_0 = patch_0.shape[-1]

            self.patchsizeX = max(patchsize_0, patchsize_1)
            self.patchsizeY = min(patchsize_0, patchsize_1)
            self.n_bands = patch_0.shape[0]
            self.SR_factor = self.patchsizeX // self.patchsizeY
        else:
            raise Exception("You need to give 2 suffixes for patches. The same as for images")

        self.lim_area_ratio = 0.9
        self.n_iter_max_ratio = 1.0
        self.sigma_blend = 1

        self.border = 16
        self.P_null = np.load(P_null_path)

    def __len__(self):
        if len(self.suffix_list) == 1:
            if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                return len(self.image_ids)
            elif 'hr' in self.suffix_list[0]:
                return self.len_factor * len(self.image_ids)
        elif len(self.suffix_list) == 2:
            return self.len_factor * len(self.image_ids)
        else:
            raise Exception(f"Invalid set of suffixes. You gave {self.suffix_list}")

    def __getitem__(self, idx):
        idx_base = idx // self.len_factor
        image_id = self.image_ids[idx_base]
        subdataset, idx_in_subds = image_id

        lr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.lr_suffix)

        if self.data_augm_type == 'None':
            if len(self.suffix_list) == 1:
                if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                    return lr_image
                elif 'hr' in self.suffix_list[0]:
                    hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                    return hr_image
            elif len(self.suffix_list) == 2:
                res_dict = {'name': image_id}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict

        elif self.data_augm_type == 'standard':
            if len(self.suffix_list) == 1 and ('lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]):
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list) == 2 and idx % self.len_factor == 0:
                res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_img')}"}
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict
            elif len(self.suffix_list) == 1 and idx % self.len_factor == 0 and 'hr' in self.suffix_list[0]:
                hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)
                return hr_image

            Fy_lim1, Fy_lim2, _, _ = self.get_Fy_fullimg_V2(
                subdataset, str(idx_in_subds), self.feas_info_patches, self.feas_app_patches,
                self.hr_suffix, self.lr_suffix, lim_area_ratio=self.lim_area_ratio,
                n_iter_max_ratio=self.n_iter_max_ratio, sigma_blend=self.sigma_blend, margin_blend=1
            )

            if len(self.suffix_list) == 1 and 'hr' in self.suffix_list[0]:
                if idx % self.len_factor == 1:
                    return Fy_lim1
                elif idx % self.len_factor == 2:
                    return Fy_lim2
                else:
                    raise Exception("Unexpected error in data augmentation.")
            elif len(self.suffix_list) == 2:
                if idx % self.len_factor == 1:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim1')}"}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 2:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim2')}"}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                return res_dict

        elif self.data_augm_type == 'standardsym':
            hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)

            if len(self.suffix_list) == 1 and ('lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]):
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list) == 2 and idx % self.len_factor == 0:
                res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_img')}"}
                res_dict[self.lr_suffix] = lr_image
                res_dict[self.hr_suffix] = hr_image
                return res_dict
            elif len(self.suffix_list) == 1 and idx % self.len_factor == 0 and 'hr' in self.suffix_list[0]:
                return hr_image

            Fy_lim1, Fy_lim2, _, _ = self.get_Fy_fullimg_V2(
                subdataset, str(idx_in_subds), self.feas_info_patches, self.feas_app_patches,
                self.hr_suffix, self.lr_suffix, lim_area_ratio=self.lim_area_ratio,
                n_iter_max_ratio=self.n_iter_max_ratio, sigma_blend=self.sigma_blend, margin_blend=1
            )

            hr_null = apply_square_op_full(self.P_null, hr_image, out_2D_shape_op=(128, 128), border=self.border)
            lim1_null = apply_square_op_full(self.P_null, Fy_lim1, out_2D_shape_op=(128, 128), border=self.border)
            lim2_null = apply_square_op_full(self.P_null, Fy_lim2, out_2D_shape_op=(128, 128), border=self.border)

            hr_sym = hr_image - 2 * hr_null
            lim1_sym = Fy_lim1 - 2 * lim1_null
            lim2_sym = Fy_lim2 - 2 * lim2_null

            if len(self.suffix_list) == 1 and 'hr' in self.suffix_list[0]:
                if idx % self.len_factor == 1:
                    return Fy_lim1
                elif idx % self.len_factor == 2:
                    return Fy_lim2
                elif idx % self.len_factor == 3:
                    return hr_sym
                elif idx % self.len_factor == 4:
                    return lim1_sym
                elif idx % self.len_factor == 5:
                    return lim2_sym
                else:
                    raise Exception("Unexpected error in data augmentation.")
            elif len(self.suffix_list) == 2:
                if idx % self.len_factor == 1:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim1')}"}
                    res_dict[self.hr_suffix] = Fy_lim1
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 2:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim2')}"}
                    res_dict[self.hr_suffix] = Fy_lim2
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 3:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_sym')}"}
                    res_dict[self.hr_suffix] = hr_sym
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 4:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim1_sym')}"}
                    res_dict[self.hr_suffix] = lim1_sym
                    res_dict[self.lr_suffix] = lr_image
                elif idx % self.len_factor == 5:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'lim2_sym')}"}
                    res_dict[self.hr_suffix] = lim2_sym
                    res_dict[self.lr_suffix] = lr_image
                else:
                    raise Exception("Unexpected error in data augmentation.")
                return res_dict

        elif self.data_augm_type == 'symetric':
            idx_base = idx // self.len_factor
            image_id = self.image_ids[idx_base]
            subdataset, idx_in_subds = image_id

            lr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.lr_suffix)
            hr_image = self.get_full_img(subdataset, idx_in_subds, suffix=self.hr_suffix)

            hr_null = apply_square_op_full(self.P_null, hr_image, out_2D_shape_op=(128, 128), border=self.border)
            hr_sym = hr_image - 2 * hr_null

            if len(self.suffix_list) == 1 and ('lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]):
                raise Exception("It makes no sense to ask for data augmentation and only ask for lr images...")
            elif len(self.suffix_list) == 1 and 'hr' in self.suffix_list[0]:
                if idx % self.len_factor == 0:
                    return hr_image
                elif idx % self.len_factor == 1:
                    return hr_sym
                else:
                    raise Exception("Unexpected error in data augmentation.")
            elif len(self.suffix_list) == 2:
                if idx % self.len_factor == 0:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_img')}"}
                    res_dict[self.lr_suffix] = lr_image
                    res_dict[self.hr_suffix] = hr_image
                elif idx % self.len_factor == 1:
                    res_dict = {'name': f"{(subdataset, idx_in_subds, 'hr_sym')}"}
                    res_dict[self.hr_suffix] = hr_sym
                else:
                    raise Exception("Unexpected error in data augmentation.")
                return res_dict

    def get_full_img(self, subdataset, img_idx, suffix):
        img_idx_list = self.imgs_idx_list[(subdataset, str(img_idx))]
        return self.recompose_image(img_idx_list, suffix, patched_shape=self.patched_shape[subdataset])

    def recompose_image(self, idx_list, suffix, patched_shape):
        patches = self.get_2D_patchgrid(idx_list, suffix, patched_shape)
        n_patches_y, n_patches_x, c, ps, ps = patches.shape
        reconstructed = patches.permute(2, 0, 3, 1, 4)
        reconstructed = reconstructed.reshape(c, n_patches_y * ps, n_patches_x * ps)
        return reconstructed

    def get_2D_patchgrid(self, idx_list, suffix, patched_shape):
        assert suffix in self.suffix_list_patches

        patches_id = [self.patch_ids[idx] for idx in idx_list]
        paths = [os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif') for patch_id in patches_id]
        patches = []
        for path in paths:
            with rasterio.open(path) as src:
                patches.append(torch.from_numpy(src.read()))

        patches = torch.stack(patches)
        c = patches.shape[1]
        ps = patches.shape[2]

        n_patches_y, n_patches_x = patched_shape
        patches = patches.view(n_patches_y, n_patches_x, c, ps, ps)

        return patches

    def amplify_mask(self, small_mask, size, patched_shape):
        nr, nc = patched_shape
        if size == 'big':
            ps = self.patchsizeX
        elif size == 'small':
            ps = self.patchsizeY
        else:
            print('Please choose a size among big or small')
            raise ValueError
        big_mask = np.zeros((nr * ps, nc * ps))

        for i in range(nr):
            for j in range(nc):
                big_mask[i * ps: (i + 1) * ps, j * ps: (j + 1) * ps] = small_mask[i, j]
        return big_mask

    def get_img_id(self, idx):
        patch_id = self.patch_ids[idx]
        splitted_patch_id = patch_id.split('_')
        idx_img = splitted_patch_id[-2]
        if len(splitted_patch_id) == 4:
            subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
        elif len(splitted_patch_id) == 3:
            subds = splitted_patch_id[0]
        return subds, idx_img

    def fill_patch_info(self, values_list, patched_shape):
        nr, nc = patched_shape
        patch_info = torch.zeros((nr, nc))
        for idx in range(len(values_list)):
            i, j = self.get_patch_position(idx, patched_shape=patched_shape)
            patch_info[int(i), int(j)] = values_list[int(idx)]
        return patch_info

    def get_patch_position(self, idx_in_img, patched_shape):
        nr, nc = patched_shape
        row = idx_in_img // nc
        col = idx_in_img % nc
        return row, col

    def get_patch_idx_img(self, position, patched_shape):
        i, j = position
        nr, nc = patched_shape
        return i * nc + j

    def get_search_shift(self, position_base, position_lim, patched_shape_base, patched_shape_lim):
        i_base, j_base = position_base
        i_lim, j_lim = position_lim
        nr_base, nc_base = patched_shape_base
        nr_lim, nc_lim = patched_shape_lim

        y_min, y_max = -min(i_base, i_lim), min(nr_base - i_base, nr_lim - i_lim)
        x_min, x_max = -min(j_base, j_lim), min(nc_base - j_base, nc_lim - j_lim)
        return y_min, y_max, x_min, x_max

    def get_same_feasible_area(self, position_base, position_lim, search_shift_lim, feas_app, base_img_id, lim_img_id, patched_shape_lim, patched_shape_base):
        i_lim, j_lim = position_lim
        i_base, j_base = position_base

        mask_same_feasible_lim = np.zeros(patched_shape_lim)
        replacement_idx = np.zeros(patched_shape_lim)
        sh_ymin, sh_ymax, sh_xmin, sh_xmax = search_shift_lim

        for i in range(sh_ymin, sh_ymax):
            for j in range(sh_xmin, sh_xmax):
                pos_1D_base = self.get_patch_idx_img((i + i_base, j + j_base), patched_shape_base)
                idx_base = self.imgs_idx_list[base_img_id][pos_1D_base]

                pos1D_lim = self.get_patch_idx_img((i + i_lim, j + j_lim), patched_shape_lim)
                idx_lim = self.imgs_idx_list[lim_img_id][pos1D_lim]

                mask_same_feasible_lim[i + i_lim, j + j_lim] = feas_app[idx_lim, idx_base]
                if mask_same_feasible_lim[i + i_lim, j + j_lim] > 0.5:
                    replacement_idx[i + i_lim, j + j_lim] = idx_lim

                if i == 0 and j == 0 and feas_app[idx_lim, idx_base] == 0 and True:
                    patch_id_base = self.patch_ids[idx_base]
                    img_base_id = self.get_img_id(int(idx_base))
                    subds_img_base = img_base_id[0]
                    position_1D_base = int(patch_id_base.split('_')[-1])
                    position_2D_base = self.get_patch_position(position_1D_base, self.patched_shape[subds_img_base])
                    print('Unexpected behaviour in the image reconstitution')
                    print(f'Looked Base position: {position_2D_base}')
                    print(f'Looked indexes: {idx_lim}, {idx_base}')

        labeled_array, num_features = label(mask_same_feasible_lim)
        component_label = labeled_array[i_lim, j_lim]

        if component_label == 0:
            return np.zeros_like(mask_same_feasible_lim, dtype=bool), np.zeros_like(mask_same_feasible_lim)

        mask_same_feasible_lim = labeled_array == component_label
        replacement_idx = replacement_idx * mask_same_feasible_lim
        return mask_same_feasible_lim, replacement_idx

    def shift_mask(self, mask, k, l, patched_shape_dst):
        shifted_res = np.zeros_like(mask)

        rows, cols = mask.shape
        nr_dst, nc_dst = patched_shape_dst

        src_row_start = max(0, -k)
        src_row_end = rows - max(0, k)
        src_col_start = max(0, -l)
        src_col_end = cols - max(0, l)

        dst_row_start = max(0, k)
        dst_row_end = rows - max(0, -k)
        dst_col_start = max(0, l)
        dst_col_end = cols - max(0, -l)

        shifted_res[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = mask[src_row_start:src_row_end, src_col_start:src_col_end]
        if rows <= nr_dst:
            shifted = np.zeros(patched_shape_dst)
            shifted[:rows, :cols] = shifted_res
        else:
            shifted = shifted_res[:nr_dst, nc_dst]
        return shifted

    def get_Fy_fullimg_idx_V2(self, subdataset, img_idx, feasible_info, feas_app, lim_area_ratio=0.6, n_iter_max_ratio=0.4):
        img_idx_list = self.imgs_idx_list[(subdataset, img_idx)]

        Fy_infos = [feasible_info[idx] for idx in img_idx_list]

        diams_Fy = torch.tensor([info[0] for info in Fy_infos])
        Fy_lim1_idx = torch.tensor([info[1][0] for info in Fy_infos])
        Fy_lim2_idx = torch.tensor([info[1][1] for info in Fy_infos])

        Fy_lim1_idx_small = self.fill_patch_info(Fy_lim1_idx, self.patched_shape[subdataset])
        Fy_lim2_idx_small = self.fill_patch_info(Fy_lim2_idx, self.patched_shape[subdataset])

        diams_Fy_small = self.fill_patch_info(diams_Fy, self.patched_shape[subdataset])

        patched_shape = self.patched_shape[subdataset]
        flat_diams = diams_Fy_small.flatten()
        sorted_values, sorted_indices = torch.sort(flat_diams, descending=True)
        nc = patched_shape[1]
        rows = sorted_indices // nc
        cols = sorted_indices % nc
        indices_2d_diams = torch.stack((rows, cols), dim=1)

        replace_lim1_idx = np.zeros(patched_shape)
        replace_lim2_idx = np.zeros(patched_shape)
        replace_lim1_order = np.zeros(patched_shape)
        replace_lim2_order = np.zeros(patched_shape)

        replaced_prop_lim1 = 0
        replaced_prop_lim2 = 0

        n_iter = 1
        n_replace = 1

        nr, nc = patched_shape
        structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        while n_iter < nr * nc * n_iter_max_ratio and replaced_prop_lim1 < lim_area_ratio and replaced_prop_lim2 < lim_area_ratio:
            maxdiam_position = indices_2d_diams[n_iter, :]
            maxdiam_position = int(maxdiam_position[0]), int(maxdiam_position[1])
            i_base, j_base = maxdiam_position

            idx_Fy_lim1 = Fy_lim1_idx_small[i_base, j_base]
            idx_Fy_lim2 = Fy_lim2_idx_small[i_base, j_base]

            lim_patch1_id = self.patch_ids[int(idx_Fy_lim1)]
            lim_patch2_id = self.patch_ids[int(idx_Fy_lim2)]

            patched_shape_base = self.patched_shape[subdataset]

            img_lim1_id = self.get_img_id(int(idx_Fy_lim1))
            subds_img_lim1 = img_lim1_id[0]
            patched_shape_lim1 = self.patched_shape[subds_img_lim1]
            patch_lim1_nb = lim_patch1_id.split('_')[-1]
            position_img_lim1 = self.get_patch_position(int(patch_lim1_nb), patched_shape=patched_shape_lim1)
            i_lim1, j_lim1 = position_img_lim1
            search_shift_lim1 = self.get_search_shift(maxdiam_position, position_img_lim1, patched_shape_base, patched_shape_lim1)
            same_feas_area_lim1, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim1, search_shift_lim1, feas_app, (subdataset, img_idx), img_lim1_id, patched_shape_lim1, patched_shape_base)
            same_feas_area_lim1_shifted, replacement_idx_shifted_lim1 = self.shift_mask(same_feas_area_lim1, i_base - i_lim1, j_base - j_lim1, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base - i_lim1, j_base - j_lim1, patched_shape_dst=patched_shape_base)

            img_lim2_id = self.get_img_id(int(idx_Fy_lim2))
            subds_img_lim2 = img_lim2_id[0]
            patched_shape_lim2 = self.patched_shape[subds_img_lim2]
            patch_lim2_nb = lim_patch2_id.split('_')[-1]
            position_img_lim2 = self.get_patch_position(int(patch_lim2_nb), patched_shape=patched_shape_lim2)
            i_lim2, j_lim2 = position_img_lim2
            search_shift_lim2 = self.get_search_shift(maxdiam_position, position_img_lim2, patched_shape_base, patched_shape_lim2)
            same_feas_area_lim2, replacement_idx = self.get_same_feasible_area(maxdiam_position, position_img_lim2, search_shift_lim2, feas_app, (subdataset, img_idx), img_lim2_id, patched_shape_lim2, patched_shape_base)
            same_feas_area_lim2_shifted, replacement_idx_shifted_lim2 = self.shift_mask(same_feas_area_lim2, i_base - i_lim2, j_base - j_lim2, patched_shape_dst=patched_shape_base), self.shift_mask(replacement_idx, i_base - i_lim2, j_base - j_lim2, patched_shape_dst=patched_shape_base)

            iterative_mask_lim1 = replace_lim1_idx > 0.5
            iterative_mask_lim2 = replace_lim2_idx > 0.5

            dilated_iterative_mask_lim1 = binary_dilation(iterative_mask_lim1, structure=structuring_element)
            dilated_iterative_mask_lim2 = binary_dilation(iterative_mask_lim2, structure=structuring_element)

            intersection_lim1 = np.logical_and(same_feas_area_lim1_shifted, dilated_iterative_mask_lim1)
            intersection_lim2 = np.logical_and(same_feas_area_lim2_shifted, dilated_iterative_mask_lim2)

            if not np.any(intersection_lim1) and not np.any(intersection_lim2):
                replace_lim1_idx = np.where(same_feas_area_lim1_shifted > 0.5, replacement_idx_shifted_lim1, replace_lim1_idx)
                replace_lim2_idx = np.where(same_feas_area_lim2_shifted > 0.5, replacement_idx_shifted_lim2, replace_lim2_idx)
                replace_lim1_order = np.where(same_feas_area_lim1_shifted > 0.5, n_replace, replace_lim1_order)
                replace_lim2_order = np.where(same_feas_area_lim2_shifted > 0.5, n_replace, replace_lim2_order)
                n_replace += 1

            n_iter += 1

            closed_mask_lim1 = binary_closing(replace_lim1_order > 0.5, structure=structuring_element)
            closed_mask_lim2 = binary_closing(replace_lim2_order > 0.5, structure=structuring_element)

            replaced_prop_lim1 = np.sum(closed_mask_lim1 > 0.5) / (patched_shape[0] * patched_shape[1])
            replaced_prop_lim2 = np.sum(closed_mask_lim2 > 0.5) / (patched_shape[0] * patched_shape[1])

        return replace_lim1_idx, replace_lim2_idx, replace_lim1_order, replace_lim2_order

    def get_Fy_fullimg_V2(self, subdataset, img_idx, feasible_info, feas_app, hr_suffix='hr_res', lr_suffix='lr_res', lim_area_ratio=0.9, n_iter_max_ratio=1.0, sigma_blend=1, margin_blend=1):
        replace_lim1_idx, replace_lim2_idx, replace_lim1_order, replace_lim2_order = self.get_Fy_fullimg_idx_V2(
            subdataset=subdataset, img_idx=img_idx, feasible_info=feasible_info, feas_app=feas_app,
            lim_area_ratio=lim_area_ratio, n_iter_max_ratio=n_iter_max_ratio
        )
        replace_lim1_idx_flat, replace_lim2_idx_flat, replace_lim1_order_flat, replace_lim2_order_flat = replace_lim1_idx.flatten(), replace_lim2_idx.flatten(), replace_lim1_order.flatten(), replace_lim2_order.flatten()
        baseimg_idx_list = np.array(self.imgs_idx_list[(subdataset, img_idx)])

        blending = True
        if not blending:
            img_lim1_idx_flat = np.where(replace_lim1_order_flat > 0.5, replace_lim1_idx_flat, baseimg_idx_list).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat > 0.5, replace_lim2_idx_flat, baseimg_idx_list).astype(int)

            Fy_lim1_img = self.recompose_image(img_lim1_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_img = self.recompose_image(img_lim2_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])
        else:
            base_imgX = self.recompose_image(baseimg_idx_list, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            img_lim1_idx_flat = np.where(replace_lim1_order_flat > 0.5, replace_lim1_idx_flat, baseimg_idx_list).astype(int)
            img_lim2_idx_flat = np.where(replace_lim2_order_flat > 0.5, replace_lim2_idx_flat, baseimg_idx_list).astype(int)

            Fy_lim1_imgY = self.recompose_image(img_lim1_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_imgY = self.recompose_image(img_lim2_idx_flat, suffix=lr_suffix, patched_shape=self.patched_shape[subdataset])

            res_mask_lim1 = replace_lim1_order > 0.5
            res_bigmask_lim1 = self.amplify_mask(res_mask_lim1, size='big', patched_shape=self.patched_shape[subdataset])

            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(res_bigmask_lim1, kernel, iterations=1)
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma=sigma_blend))

            Fy_lim1_img_noblend = self.recompose_image(img_lim1_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim1_img = blurred_mask * Fy_lim1_img_noblend + (1 - blurred_mask) * base_imgX

            res_mask_lim2 = replace_lim2_order > 0.5
            res_bigmask_lim2 = self.amplify_mask(res_mask_lim2, size='big', patched_shape=self.patched_shape[subdataset])

            eroded_mask = cv2.erode(res_bigmask_lim2, kernel, iterations=1)
            blurred_mask = torch.tensor(gaussian_filter(eroded_mask, sigma=sigma_blend))

            Fy_lim2_img_noblend = self.recompose_image(img_lim2_idx_flat, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
            Fy_lim2_img = blurred_mask * Fy_lim2_img_noblend + (1 - blurred_mask) * base_imgX

        return Fy_lim1_img, Fy_lim2_img, Fy_lim1_imgY, Fy_lim2_imgY

class SRDataset(Dataset):
    """
    Obsolete dataset class for super-resolution image patches.
    """
    def __init__(
        self,
        folder_path,
        suffixes=('lr', 'hr'),
        patched_shapes={'naip': (40, 40), 'spain_crops': (42, 42), 'spain_urban': (42, 42), 'spot': (42, 42)}
    ):
        def get_img_id(patch_id):
            splitted_patch_id = patch_id.split('_')
            idx_img = splitted_patch_id[-2]
            if len(splitted_patch_id) == 4:
                subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
            elif len(splitted_patch_id) == 3:
                subds = splitted_patch_id[0]
            return subds, idx_img

        self.folder_path = folder_path
        self.suffix_list = suffixes if not isinstance(suffixes, str) else [suffixes]

        self.file_list = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list[0]}.tif')))

        self.patched_shape = patched_shapes

        self.patch_ids = [os.path.basename(f).replace(f'_{self.suffix_list[0]}.tif', '') for f in self.file_list]

        self.image_ids = list(set([get_img_id(p_id) for p_id in self.patch_ids]))

        imgs_idx_list = {}

        for img_id in self.image_ids:
            subds, n_img = img_id

            img_idx_list = [(i, self.patch_ids[i].split('_')[-1]) for i in range(len(self.patch_ids)) if (f'{subds}_{n_img}_' in self.patch_ids[i])]
            img_idx_list = sorted(
                img_idx_list,
                key=lambda x: int(x[1])
            )

            img_idx_list = [int(x[0]) for x in img_idx_list]

            imgs_idx_list[img_id] = img_idx_list

        self.imgs_idx_list = imgs_idx_list

        self.patched_shape = patched_shapes

        if len(self.suffix_list) == 2:
            self.file_list1 = sorted(glob.glob(os.path.join(folder_path, f'*_{self.suffix_list[1]}.tif')))

            with rasterio.open(self.file_list1[0]) as src:
                patch_1 = torch.from_numpy(src.read())
                patchsize_1 = patch_1.shape[-1]
            with rasterio.open(self.file_list[0]) as src:
                patch_0 = torch.from_numpy(src.read())
                patchsize_0 = patch_0.shape[-1]

            self.patchsizeX = max(patchsize_0, patchsize_1)
            self.patchsizeY = min(patchsize_0, patchsize_1)

            self.n_bands = patch_0.shape[0]

            self.SR_factor = self.patchsizeX // self.patchsizeY
        elif len(self.suffix_list) == 1:
            if 'lr' in self.suffix_list[0] or 'DSHR' in self.suffix_list[0]:
                with rasterio.open(self.file_list[0]) as src:
                    patch_0 = torch.from_numpy(src.read())
                    self.patchsizeY = patch_0.shape[-1]
                    self.n_bands = patch_0.shape[0]
            elif 'hr' in self.suffix_list[0]:
                with rasterio.open(self.file_list[0]) as src:
                    patch_0 = torch.from_numpy(src.read())
                    self.patchsizeX = patch_0.shape[-1]
                    self.n_bands = patch_0.shape[0]
            else:
                print('lr or hr not in the suffix names. Please add one of them.')
                print(f'Your suffixes are {suffixes}')
        else:
            print('Only 1 or 2 suffixes are allowed')
            print(f'You gave suffixes = {suffixes}')

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]

        if len(self.suffix_list) == 1:
            img_path = os.path.join(self.folder_path, f'{patch_id}_{self.suffix_list[0]}.tif')

            with rasterio.open(img_path) as src:
                img = torch.from_numpy(src.read())

            return img
        else:
            result_dict = {"name": patch_id}

            for suffix in self.suffix_list:
                img_path = os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif')

                with rasterio.open(img_path) as src:
                    result_dict[suffix] = torch.from_numpy(src.read())

            return result_dict

    def get_img_id(self, idx):
        patch_id = self.patch_ids[idx]
        splitted_patch_id = patch_id.split('_')
        idx_img = splitted_patch_id[-2]
        if len(splitted_patch_id) == 4:
            subds = f'{splitted_patch_id[0]}_{splitted_patch_id[1]}'
        elif len(splitted_patch_id) == 3:
            subds = splitted_patch_id[0]
        return subds, idx_img

    def fill_patch_uniform_img(self, values_list, patchsize, patched_shape):
        unif_patches = []
        for x in values_list:
            unif_patches.append(x * torch.ones((patchsize, patchsize)))

        unif_patches = torch.stack(unif_patches)

        n_patches_y, n_patches_x = patched_shape

        reconstructed = unif_patches.view(n_patches_y, n_patches_x, patchsize, patchsize)
        reconstructed = reconstructed.permute(0, 2, 1, 3).contiguous()
        reconstructed = reconstructed.view(n_patches_y * patchsize, n_patches_x * patchsize)

        return reconstructed

    def fill_patch_info(self, values_list, patched_shape):
        nr, nc = patched_shape
        patch_info = torch.zeros((nr, nc))
        for idx in range(len(values_list)):
            i, j = self.get_patch_position(idx, patched_shape=patched_shape)
            patch_info[int(i), int(j)] = values_list[int(idx)]
        return patch_info

    def get_patch_position(self, idx_in_img, patched_shape):
        nr, nc = patched_shape
        row = idx_in_img // nc
        col = idx_in_img % nc
        return row, col

    def get_patch_idx_img(self, position, patched_shape):
        i, j = position
        nr, nc = patched_shape
        return i * nc + j

    def recompose_image(self, idx_list, suffix, patched_shape):
        patches = self.get_2D_patchgrid(idx_list, suffix, patched_shape)

        n_patches_y, n_patches_x, c, ps, ps = patches.shape
        reconstructed = patches.permute(2, 0, 3, 1, 4)
        reconstructed = reconstructed.reshape(c, n_patches_y * ps, n_patches_x * ps)

        return reconstructed

    def get_2D_patchgrid(self, idx_list, suffix, patched_shape):
        assert suffix in self.suffix_list

        patches_id = [self.patch_ids[idx] for idx in idx_list]
        paths = [os.path.join(self.folder_path, f'{patch_id}_{suffix}.tif') for patch_id in patches_id]
        patches = []
        for path in paths:
            with rasterio.open(path) as src:
                patches.append(torch.from_numpy(src.read()))

        patches = torch.stack(patches)
        c = patches.shape[1]
        ps = patches.shape[2]

        n_patches_y, n_patches_x = patched_shape

        patches = patches.view(n_patches_y, n_patches_x, c, ps, ps)

        return patches

    def generate_mask(self, patchsize):
        mask = np.ones((patchsize, patchsize), dtype=np.uint8) * 255
        return mask

    def seamless_blending(self, patch1, patch2, mask, center):
        """
        Apply Poisson blending using OpenCV's seamlessClone for RGB images.
        - patch1: The patch to blend (source).
        - patch2: The base image (destination).
        - mask: Binary mask indicating where blending happens.
        - center: Center point for the cloning (usually the center of patch2).
        """
        n_bands = patch1.shape[0]

        patch1_ = patch1.transpose(1, 2, 0)
        patch2_ = patch2.transpose(1, 2, 0)
        result_ = np.zeros_like(patch2_)

        result_ = cv2.seamlessClone(patch1_, patch2_, mask, center, cv2.NORMAL_CLONE)

        result = result_.transpose(2, 0, 1)
        return result

    def stitch_patches(self, patches, patchsize, grid_size, base_image, pixel_range=3000):
        """
        Stitch the nrxnc patches together using Poisson blending for multiple bands images.
        """
        patches_ = patches * 255 / pixel_range
        patches_ = patches_.astype(np.uint8)

        base_image_ = base_image * 255 / pixel_range
        base_image_ = base_image_.astype(np.uint8)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                patch = patches_[i, j]

                x_offset = j * patchsize
                y_offset = i * patchsize
                patch_position = (x_offset + patchsize // 2, y_offset + patchsize // 2)

                mask = self.generate_mask(patchsize)

                base_image_ = self.seamless_blending(patch, base_image_, mask, patch_position)

        base_image_ = base_image_.astype(np.float32)
        return base_image_ * pixel_range / 255

    def get_Fy_lim_fullimg_poissonblending(self, subdataset, img_idx, feasible_info, hr_suffix='hr_res'):
        img_idx_list = [(i, self.patch_ids[i].split('_')[-1]) for i in range(len(self.patch_ids)) if (f'{subdataset}_{img_idx}_' in self.patch_ids[i])]

        img_idx_list = sorted(
            img_idx_list,
            key=lambda x: int(x[1])
        )
        img_idx_list = [int(x[0]) for x in img_idx_list]

        Fy_infos = [feasible_info[idx] for idx in img_idx_list]
        Fy_lim1_idx = [info[1][0] for info in Fy_infos]
        Fy_lim2_idx = [info[1][1] for info in Fy_infos]

        Fy_lim1_patchgrid = self.get_2D_patchgrid(Fy_lim1_idx, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])
        Fy_lim2_patchgrid = self.get_2D_patchgrid(Fy_lim2_idx, suffix=hr_suffix, patched_shape=self.patched_shape[subdataset])

        nr, nc = Fy_lim1_patchgrid.shape[:2]

        base_image = self.get_full_img(subdataset, img_idx, hr_suffix)
        Fy_lim1_img = torch.tensor(self.stitch_patches(np.array(Fy_lim1_patchgrid), patchsize=self.patchsizeX, grid_size=(nr, nc), base_image=np.array(base_image)))
        Fy_lim2_img = torch.tensor(self.stitch_patches(np.array(Fy_lim2_patchgrid), patchsize=self.patchsizeX, grid_size=(nr, nc), base_image=np.array(base_image)))

        Fy_cards = [info[2] for info in Fy_infos]
        cards = self.fill_patch_uniform_img(Fy_cards, patchsize=self.patchsizeX, patched_shape=self.patched_shape[subdataset])

        return Fy_lim1_img, Fy_lim2_img, cards
