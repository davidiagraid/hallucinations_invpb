import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
import torch
from scipy.ndimage import gaussian_filter, binary_closing, label
from skimage.measure import block_reduce
from torch.utils.data import Dataset
import cv2
from typing import Optional
from abc import ABC, abstractmethod

class ImageOnlyDataset(Dataset):
    """A PyTorch Dataset wrapper that returns only images from a given dataset.

    Attributes:
        dataset: The underlying dataset to wrap.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns only the image from the dataset at the given index."""
        image, _ = self.dataset[idx]
        return image

class MatrixOpCalculator:
    """Computes and analyzes sparse matrices for linear operators.

    Attributes:
        n_in (int): Input dimension.
        n_out (int): Output dimension.
        operator (callable): The linear operator to represent as a sparse matrix.
        num_workers (int): Number of parallel workers for matrix construction.
        singular_threshold_ratio (float): Threshold for singular value decomposition.
    """
    def __init__(
        self,
        n_in: int,
        n_out: int,
        Operator: callable,
        num_workers: Optional[int] = None,
        singular_threshold_ratio: float = 0.001
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.operator = Operator
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.singular_threshold_ratio = singular_threshold_ratio

    def compute_column(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the i-th column of the sparse matrix.

        Args:
            i: Column index.

        Returns:
            Tuple of (row_indices, column_indices, data) for the i-th column.
        """
        A = self.operator
        e_i = np.zeros(self.n_in)
        e_i[i] = 1.0
        col = A(e_i)

        # Find non-zero entries
        row_idx = np.nonzero(col)[0]
        data = col[row_idx]
        col_idx = np.full_like(row_idx, i)

        return row_idx, col_idx, data

    def build_sparse_matrix_parallel(self) -> coo_matrix:
        """Builds the sparse matrix in parallel using multiprocessing.

        Returns:
            The sparse matrix in COO format.
        """
        with Pool(self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.compute_column, range(self.n_in)), total=self.n_in))

        row_indices = []
        col_indices = []
        data = []

        for r, c, d in results:
            row_indices.extend(r)
            col_indices.extend(c)
            data.extend(d)

        return coo_matrix((data, (row_indices, col_indices)), shape=(self.n_out, self.n_in)).tocsc()

    def get_range_space_basis(self, A_sparse: coo_matrix, sigma_threshold_ratio: float = 0.001) -> np.ndarray:
        """Computes the basis for the range space of the sparse matrix.

        Args:
            A_sparse: The sparse matrix.
            sigma_threshold_ratio: Threshold for singular values.

        Returns:
            The basis for the range space.
        """
        p, q = A_sparse.shape
        kmax = int(min(p, q) - 1)
        umat, sing, vt = svds(A_sparse, k=kmax)
        threshold = sigma_threshold_ratio * np.max(sing)
        return vt[sing >= threshold].T

    def make_null_projection_operator(self, range_basis: np.ndarray) -> np.ndarray:
        """Constructs the projection operator onto the null space.

        Args:
            range_basis: The basis for the range space.

        Returns:
            The projection operator.
        """
        return np.eye(range_basis.shape[0]) - range_basis.dot(range_basis.T)

def downsample_gaussian_meanpool(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsamples a 2D array using Gaussian blur followed by mean pooling.

    Args:
        x: Input image of shape (H, W).
        factor: Downsampling factor.

    Returns:
        Downsampled image of shape (H//factor, W//factor).
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)
    else:
        x = x.astype(float)

    if x.ndim != 2:
        raise ValueError("Input must be 2D (H, W)")

    sigma = factor
    x_blur = gaussian_filter(x, sigma=sigma, mode='reflect')
    return block_reduce(x_blur, block_size=(factor, factor), func=np.mean)

def DS_op_3_28_10(x: np.ndarray) -> np.ndarray:
    """Downsamples a 28x28 image by a factor of 3 using Gaussian mean pooling.

    Args:
        x: Input image as a flattened array.

    Returns:
        Downsampled image as a flattened array.
    """
    x = x.reshape(28, 28)
    y = downsample_gaussian_meanpool(x, factor=3)
    return np.asarray(y).flatten()

class DistanceMetric(ABC):
    """Abstract base class for computing distance metrics between tensors.

    Attributes:
        method (str): Distance computation method ("pixel", "patch", or "image").
        patch_size (int): Patch size for patch-based methods.
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
    """
    def __init__(
        self,
        method: str,
        patch_size: int,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ):
        self.method = method
        self.patch_size = patch_size
        self.kwargs = kwargs
        self.axis: int = 0
        self.x = x
        self.y = y

    @staticmethod
    def do_square(tensor: torch.Tensor, patch_size: Optional[int] = 32) -> torch.Tensor:
        """Splits a tensor into square patches.

        Args:
            tensor: Input tensor of shape (C, H, W).
            patch_size: Size of each patch.

        Returns:
            Tensor of patches.
        """
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("The tensor must be square.")

        xdim, ydim = tensor.shape[1], tensor.shape[2]
        minimages_x = int(torch.ceil(torch.tensor(xdim / patch_size)))
        minimages_y = int(torch.ceil(torch.tensor(ydim / patch_size)))

        pad_x_01 = int((minimages_x * patch_size - xdim) // 2)
        pad_x_02 = int((minimages_x * patch_size - xdim) - pad_x_01)
        pad_y_01 = int((minimages_y * patch_size - ydim) // 2)
        pad_y_02 = int((minimages_y * patch_size - ydim) - pad_y_01)

        padded_tensor = torch.nn.functional.pad(tensor, (pad_x_01, pad_x_02, pad_y_01, pad_y_02))
        patches = padded_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        return patches.permute(1, 2, 0, 3, 4)

    @abstractmethod
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def compute_image(self) -> torch.Tensor:
        return self._compute_image(self.x, self.y)

    def compute_patch(self) -> torch.Tensor:
        x_batched = self.do_square(self.x, self.patch_size)
        y_batched = self.do_square(self.y, self.patch_size)
        metric_result = torch.zeros(x_batched.shape[:2])
        xrange, yrange = x_batched.shape[0:2]
        for x_index in range(xrange):
            for y_index in range(yrange):
                x_batch = x_batched[x_index, y_index]
                y_batch = y_batched[x_index, y_index]
                metric_result[x_index, y_index] = self._compute_image(x_batch, y_batch)
        return torch.nn.functional.interpolate(metric_result[None, None], size=self.x.shape[-2:], mode="nearest").squeeze()

    def compute_pixel(self) -> torch.Tensor:
        return self._compute_pixel(self.x, self.y)

    def compute(self) -> torch.Tensor:
        if self.method == "pixel":
            return self.compute_pixel()
        elif self.method == "image":
            return self.compute_image()
        elif self.method == "patch":
            return self.compute_patch()
        else:
            raise ValueError("Invalid method.")

class L1(DistanceMetric):
    """Computes the L1 distance between two tensors."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, method: str = "image", patch_size: int = 32):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y), axis=0)

class L2(DistanceMetric):
    """Computes the L2 distance between two tensors."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, method: str = "image", patch_size: int = 32):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2) ** 0.5

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2, axis=0) ** 0.5

class LP(DistanceMetric):
    """Computes the Lp distance between two tensors."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, p: int, method: str = "image", patch_size: int = 32):
        self.p = p
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.nanmean(torch.abs(x - y) ** self.p)) ** (1 / self.p)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.nanmean(torch.abs(x - y) ** self.p, axis=0)) ** (1 / self.p)
    

def auto_labeling_MNIST(img1, img2, threshold_bin = 0, min_size_filter = 1, dist_merge = 4, min_hall_size = 4):
    # First binarize both images
    img1 = np.array(img1)
    img2 = np.array(img2)

    img1_bin = (img1>threshold_bin).astype(float)
    img2_bin = (img2>threshold_bin).astype(float)



    #Diff between binarized
    delta_bin = np.abs(img1_bin-img2_bin)
    filtered_binary = np.zeros_like(delta_bin)

    # Eliminate components with less than 2 pixels
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_array, num_features = label(delta_bin, structure=structure)

    for component in range(1, num_features + 1):
        component_mask = labeled_array == component
        if np.sum(component_mask) > min_size_filter:
            filtered_binary[component_mask] = 1


    #Closing of the image in order to merge together compnents that are close enough to each other
    filtered_binary = filtered_binary.astype(bool)
    kernel_size = max(1, dist_merge // 2)
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    merged = binary_closing(filtered_binary, structure=structure)
    merged = binary_closing(merged, structure=structure)
    #merged = merged.astype(np.uint8)
    
    if False:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img1)

        axes[1].imshow(img1_bin)
        comp = ImgComparator(fig, axes)
        plt.show()

    # Tliminate again too small components
    filtered_binary = np.zeros_like(delta_bin)
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_array, num_features = label(merged, structure=structure)

    for component in range(1, num_features + 1):
        component_mask = labeled_array == component
        if np.sum(component_mask) > min_hall_size:
            filtered_binary[component_mask] = 1

    # Binarize again
    return filtered_binary.astype(float)

def get_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    agg_method: str,
    patch_size: int = 32,
) -> torch.Tensor:
    """Computes the distance between two tensors using the specified method and aggregation.

    Args:
        x: First input tensor.
        y: Second input tensor.
        method: Distance method ("l1", "l2", "lp").
        agg_method: Aggregation method ("pixel", "image", "patch").
        patch_size: Patch size for patch-based aggregation.

    Returns:
        The computed distance.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("The number of channels in x and y must be the same.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("The height of x and y must be the same.")

    if method == "l1":
        distance_fn = L1(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "l2":
        distance_fn = L2(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method[:2] == "lp":
        p = int(method[2:])
        distance_fn = LP(x=x, y=y, p=p, method=agg_method, patch_size=patch_size)
    else:
        raise ValueError("No valid distance method.")

    return distance_fn.compute()