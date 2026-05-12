from functools import partial
import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset
from matplotlib import pyplot as plt
from pdb import set_trace
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from torchvision import datasets, transforms

scale_factor = 3

def downsample_gaussian_meanpool(x, factor):
    """
    Downsample a 2D array (H, W) using Gaussian blur + mean pooling.
    Uses SciPy / skimage functions instead of PyTorch.
    
    Parameters:
    -----------
    x : 2D np.array
        Input image of shape (H, W)
    factor : int
        Downsampling factor
        
    Returns:
    --------
    x_down : 2D np.array
        Downsampled image of shape (H//factor, W//factor)
    """
    # Convert input to NumPy array if needed
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)
    else:
        x = x.astype(float)

    if x.ndim != 2:
        raise ValueError("Input must be 2D (H, W)")

    # Apply Gaussian blur
    sigma = factor  # similar to previous PyTorch version
    x_blur = gaussian_filter(x, sigma=sigma, mode='reflect')

    # Apply mean pooling (block reduce)
    x_down = block_reduce(x_blur, block_size=(factor, factor), func=np.mean)

    return x_down

def DS_operator_formatrix(img, intermediate_op):
    img_padded = F.pad(img.unsqueeze(0), (4,4,4,4), mode = 'reflect').squeeze()
    DS_img = intermediate_op(img_padded)

    return DS_img


intermediate_operator_formatrix = partial(downsample_gaussian_meanpool, factor = scale_factor)
DS_for_matrix = partial(DS_operator_formatrix, intermediate_op = intermediate_operator_formatrix)

def poisson_noise(img, target_norm):
    h,w = img.shape[-2:]
    N = h*w
    img_interm = np.array(img-img.min()+ 0.001)
    alpha_opt = (target_norm**2)/(N*img_interm.mean())
    noisy_img = torch.tensor(np.random.poisson(img_interm/alpha_opt)) *alpha_opt + img.min() - 0.001
    return torch.tensor(noisy_img)

def gaussian_noise(img, target_norm):
    h,w = img.shape[-2:]
    N = h*w
    img = torch.tensor(img)
    sigma = target_norm/(N**0.5)
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    return noisy_img

transform = transforms.Compose([
            transforms.ToTensor(),       
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.to(torch.float32)),
            ])

def transform_HRDS(noise_type = None,noise_level = 0.3):
    if noise_type == None or noise_level == 0:
        return transform_HRDS
    elif noise_type == 'gauss':
        transform_HRDS_gauss = transforms.Compose([
        transforms.ToTensor(),  # (1, H, W)
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.squeeze(0)),   # (H, W)
        transforms.Lambda(lambda x: DS_for_matrix(x)),
        transforms.Lambda(lambda x: gaussian_noise(x, target_norm=noise_level)),
        transforms.Lambda(lambda x: torch.tensor(np.expand_dims(x, axis = 0).astype(np.float32))),
        transforms.Resize((28, 28))
        ])
        return transform_HRDS_gauss
    elif noise_type == 'poisson':
        transform_HRDS_poisson = transforms.Compose([
            transforms.ToTensor(),  # (1, H, W)
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.squeeze(0)),   # (H, W)
            transforms.Lambda(lambda x: DS_for_matrix(x)),
            transforms.Lambda(lambda x: poisson_noise(x, target_norm=noise_level)),
            transforms.Lambda(lambda x: torch.tensor(np.expand_dims(x, axis = 0).astype(np.float32))),
            transforms.Resize((28, 28))
            ])
        return transform_HRDS_poisson

class PairedMNIST(datasets.MNIST):
    def __init__(self, root,transform_Y = transform_HRDS(None, 0), download=True):
        super().__init__(
            root=root,
            train=True,
            transform=None,      # IMPORTANT: disable default transform
            download=download
        )
        self.transform_Y = transform_Y


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Convert to PIL (MNIST stores uint8 tensors)
        img = img.numpy()

        img_HR = transform(img)
        img_LR = self.transform_Y(img)


        return img_LR,img_HR
