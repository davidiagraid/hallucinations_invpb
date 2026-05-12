from torch.utils.data import Dataset
import h5py
import numpy as np

class Brain_dataset(Dataset):
    """
    A PyTorch Dataset for brain MRI data, supporting both k-space and image domain data.

    Attributes:
        data_info (dict): Dictionary containing information about each data file.
        slices_idx (list): List of slice indices to include in the dataset.
        k_mask (np.ndarray): Mask to apply to k-space data.
        data_folder (str): Path to the folder containing data files.
        shape_x (tuple): Target shape for image data (default: (320, 320)).
        type (str): Type of data to load ('k_mes' for k-space, 'x' for image).
        x_ROImask (np.ndarray): Mask to apply to image domain data (default: ones).
        dataset (list): List of tuples (file, slice, coil) representing dataset items.
    """
    def __init__(self, data_folder, data_info, slices_idx, k_mask, type, shape_x=(320, 320), x_ROImask=np.ones((320, 320))):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            data_info (dict): Dictionary with 'fileorder' and 'files_info' keys.
            slices_idx (list): List of slice indices to include.
            k_mask (np.ndarray): Mask for k-space data.
            type (str): Type of data to load ('k_mes' or 'x').
            shape_x (tuple): Target shape for image data.
            x_ROImask (np.ndarray): Mask for image domain data.
        """
        assert type in ['k_mes', 'x'], "Type must be either 'k_mes' or 'x'"

        self.data_info = data_info
        self.slices_idx = slices_idx
        self.k_mask = k_mask
        self.data_folder = data_folder
        self.shape_x = shape_x
        self.type = type  # among 'k_mes' and 'x'
        self.x_ROImask = x_ROImask

        dataset = []
        for file in data_info['fileorder']:
            n_s, n_coils = data_info['files_info'][file]
            for slice in slices_idx:
                for i in range(n_coils):
                    dataset.append((file, slice, i))
        self.dataset = dataset

    def __len__(self):
        """
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (file, slice, coil) tuple for the item at index idx.
        """
        return self.dataset[idx]

class Brain_dataset_preloaded(Dataset):
    """
    A PyTorch Dataset for preloaded brain MRI data, supporting both k-space and image domain data.

    Attributes:
        data_info (dict): Dictionary containing information about each data file.
        slices_idx (list): List of slice indices to include in the dataset.
        k_mask (np.ndarray): Mask to apply to k-space data.
        data_folder (str): Path to the folder containing data files.
        shape_x (tuple): Target shape for image data (default: (320, 320)).
        type (str): Type of data to load ('k_mes' for k-space, 'x' for image).
        x_ROImask (np.ndarray): Mask to apply to image domain data (default: ones).
        dataset (list): List of tuples (file, slice, coil) representing dataset items.
    """
    def __init__(self, data_folder, data_info, slices_idx, k_mask, type, shape_x=(320, 320), x_ROImask=np.ones((320, 320))):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            data_info (dict): Dictionary with 'fileorder' and 'files_info' keys.
            slices_idx (list): List of slice indices to include.
            k_mask (np.ndarray): Mask for k-space data.
            type (str): Type of data to load ('k_mes' or 'x').
            shape_x (tuple): Target shape for image data.
            x_ROImask (np.ndarray): Mask for image domain data.
        """
        assert type in ['k_mes', 'x'], "Type must be either 'k_mes' or 'x'"

        self.data_info = data_info
        self.slices_idx = slices_idx
        self.k_mask = k_mask
        self.data_folder = data_folder
        self.shape_x = shape_x
        self.type = type  # among 'k_mes' and 'x'
        self.x_ROImask = x_ROImask

        dataset = []
        for file in data_info['fileorder']:
            n_s, n_coils = data_info['files_info'][file]
            for slice in slices_idx:
                for i in range(n_coils):
                    dataset.append((file, slice, i))
        self.dataset = dataset

    def __len__(self):
        """
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            np.ndarray: Loaded and masked data for the item at index idx.
        """
        file, slice, i = self.dataset[idx]
        if self.type == 'k_mes':
            key = 'kspace'
            mask = self.k_mask
        elif self.type == 'x':
            key = 'reconstruction_rss'
            mask = self.x_ROImask

        with h5py.File(f'{self.data_folder}/{file}', "r") as f:
            data = f[key][slice, i]
            data *= mask
        return data

class BrainSC_dataset_preloaded(Dataset):
    """
    A PyTorch Dataset for preloaded single-coil brain MRI data, supporting both k-space and image domain data.

    Attributes:
        data_info (dict): Dictionary containing information about each data file.
        slices_idx (list): List of slice indices to include in the dataset.
        k_mask (np.ndarray): Mask to apply to k-space data.
        data_folder (str): Path to the folder containing data files.
        shape_x (tuple): Target shape for image data (default: (320, 320)).
        type (str): Type of data to load ('k_mes' for k-space, 'x' for image).
        x_ROImask (np.ndarray): Mask to apply to image domain data (default: ones).
        dataset (list): List of tuples (file, slice) representing dataset items.
    """
    def __init__(self, data_folder, data_info, slices_idx, k_mask, type, shape_x=(320, 320), x_ROImask=np.ones((320, 320))):
        """
        Args:
            data_folder (str): Path to the folder containing data files.
            data_info (dict): Dictionary with 'fileorder' and 'files_info' keys.
            slices_idx (list): List of slice indices to include.
            k_mask (np.ndarray): Mask for k-space data.
            type (str): Type of data to load ('k_mes' or 'x').
            shape_x (tuple): Target shape for image data.
            x_ROImask (np.ndarray): Mask for image domain data.
        """
        assert type in ['k_mes', 'x'], "Type must be either 'k_mes' or 'x'"

        self.data_info = data_info
        self.slices_idx = slices_idx
        self.k_mask = k_mask
        self.data_folder = data_folder
        self.shape_x = shape_x
        self.type = type  # among 'k_mes' and 'x'
        self.x_ROImask = x_ROImask

        dataset = []
        for file in data_info['fileorder']:
            n_s, n_coils = data_info['files_info'][file]
            for slice in slices_idx:
                dataset.append((file, slice))
        self.dataset = dataset

    def __len__(self):
        """
        Returns:
            int: Number of items in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            np.ndarray: Loaded and masked data for the item at index idx.
        """
        file, slice = self.dataset[idx]
        if self.type == 'k_mes':
            key = 'kspace'
            mask = self.k_mask
        elif self.type == 'x':
            key = 'reconstruction_rss'
            mask = self.x_ROImask

        with h5py.File(f'{self.data_folder}/{file}', "r") as f:
            data = f[key][slice]
            data *= mask
        return data

    def get_info(self, idx):
        """
        Args:
            idx (int): Index of the item.

        Returns:
            tuple: (file, slice) tuple for the item at index idx.
        """
        file, slice = self.dataset[idx]
        return file, slice

    def get_idx(self, infos):
        """
        Args:
            infos (tuple): (file, slice) tuple to find.

        Returns:
            int: Index of the item, or None if not found.
        """
        try:
            x, y = infos
            idx = self.dataset.index((x, y))
            return idx
        except Exception as e:
            return None


    