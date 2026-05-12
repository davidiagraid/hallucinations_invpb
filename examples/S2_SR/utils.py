import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
import os
from tqdm import tqdm
from typing import List, Optional, Union, Tuple
from multiprocessing import Pool, cpu_count
from scipy.sparse.linalg import svds
from scipy.sparse import coo_matrix
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib import pyplot as plt
import opensr_model
import cv2 as cv

def apply_polygon_mask(img: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
    """
    Applies a polygon mask to an image, setting pixels inside any polygon to 0.

    Args:
        img: Input image (H, W, ...).
        polygons: List of polygons, each as a numpy array of vertices.

    Returns:
        Masked image as a numpy array.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    for poly in polygons:
        path = Path(poly)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
        inside = path.contains_points(coords).reshape(h, w)
        mask |= inside

    img_masked = np.ones_like(img, dtype=int)
    img_masked[mask] = 0

    return img_masked

class ImgPastingS2():
    def __init__(self, fig, img_ids, dataset_path):
        self.fig = fig
        self.axes = fig.get_axes()
        self.img_ids = img_ids
        self.dataset_path = dataset_path

        self.verts = []
        self.lines = [ax.plot([], [], marker='o', color='cyan', linestyle='-')[0] for ax in [self.axes[1]]]
        self.polygons_list = []
 
        self.img_base_currentid_idx = 0
        self.img_det_currentid_idx = 1

        img_base_currentid = self.img_ids[self.img_base_currentid_idx]
        img_det_currentid = self.img_ids[self.img_det_currentid_idx]

        self.base_img_id = img_base_currentid

        img_base_path = os.path.join(dataset_path, img_base_currentid[0], img_base_currentid[1], 'hr_res.tif')
        img_det_path = os.path.join(dataset_path, img_det_currentid[0], img_det_currentid[1], 'hr_res.tif')

        with rasterio.open(img_base_path) as src:
            img_base = torch.from_numpy(src.read())
            self.img_base = self.axes[0].imshow(img_base.permute(1,2,0)/3000)
            self.axes[0].set_title(f'Base image : {img_base_currentid[0]}  {img_base_currentid[1]}')
            self.axes[0].set_xlim(0, img_base.shape[2])
            self.axes[0].set_ylim(0, img_base.shape[1])

        with rasterio.open(img_det_path) as src:
            img_det = torch.from_numpy(src.read())
            self.img_det = self.axes[1].imshow(img_det.permute(1,2,0)/3000)
            self.axes[1].set_title(f'Pick a detail from : {img_det_currentid[0]}  {img_det_currentid[1]}')
        self.fig.canvas.draw()


        # overlay (grayscale)
        self.overlay = np.zeros((128, 128,3)) 

        # initial overlay position (bottom-left corner)
        x0, y0 = 50, 50
        h, w = self.overlay.shape[:2]

        self.h = h
        self.w = w

        # store center coordinates
        self.center_x = x0 + w / 2
        self.center_y = y0 + h / 2

        self.overlay_artist = self.axes[0].imshow(
            self.overlay,
            extent=[x0, x0 + w, y0, y0 + h],
            alpha=0.9,
            cmap='gray', 
            origin = 'upper'
        )

        #self.overlay_artist.set_extent([0, img_base.shape[2], 0, img_base.shape[1]])
        self.img_base.set_extent([0, img_base.shape[2], 0, img_base.shape[1]])


        # connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def update_overlay_artist(self, new_overlay):
        h, w = new_overlay.shape[:2]
        self.overlay = new_overlay

        self.h = h
        self.w = w

        self.overlay_artist.set_data(self.overlay)
        self.overlay_artist.set_extent([
            self.center_x - self.w / 2,
            self.center_x + self.w / 2,
            self.center_y - self.h / 2,
            self.center_y + self.h / 2
            ])
        
        self.overlay_artist.set_cmap('gray')
        self.overlay_artist.set_clim(
                vmin=float(new_overlay.min()),
                vmax=float(new_overlay.max())
            )
        #self.overlay_artist.set_origin('upper')
        self.fig.canvas.draw_idle()
    
    def init_overlay(self):
        self.update_overlay_artist(np.ones((128,128,3)))
    
    def get_overlay_array(self):
        """Return a numpy array of shape (H_base, W_base) with overlay positioned, rest zeros"""
        H_base, W_base, bands = self.img_base.get_array().shape
        arr = np.zeros((H_base, W_base, bands), dtype=self.overlay.dtype)

        # current overlay size
        h_cur = int(round(self.h))
        w_cur = int(round(self.w))

    
        # convert current overlay extent to integer pixel indices
        x0 = int(round(self.center_x - w_cur / 2))
        y0 = int(round(self.center_y - h_cur / 2))
        x1 = x0 + w_cur
        y1 = y0 + h_cur

        # clamp to base image boundaries
        x0_clamped = max(0, x0)
        y0_clamped = max(0, y0)
        x1_clamped = min(W_base, x1)
        y1_clamped = min(H_base, y1)

        # compute slice indices in overlay_resized
        overlay_x0 = x0_clamped - x0
        overlay_y0 = y0_clamped - y0
        overlay_x1 = overlay_x0 + (x1_clamped - x0_clamped)
        overlay_y1 = overlay_y0 + (y1_clamped - y0_clamped)
        
        arr[H_base-y1_clamped:W_base-y0_clamped, x0_clamped:x1_clamped] = \
            self.overlay[overlay_y0:overlay_y1, overlay_x0:overlay_x1]

        return arr

    # First make an interface to navigate through the images
    def on_key_press(self, event):
        if event.key == 'right': # base image

            # Update the image base id and idx
            self.img_base_currentid_idx = (self.img_base_currentid_idx+1)%len(self.img_ids)
            img_base_currentid = self.img_ids[self.img_base_currentid_idx]
            self.base_img_id = img_base_currentid

            # retrieve the corresponding image
            img_base_path = os.path.join(self.dataset_path, img_base_currentid[0], img_base_currentid[1], 'hr_res.tif')
            with rasterio.open(img_base_path) as src:
                img_base = torch.from_numpy(src.read())

            # Update the axes[0] and the title
            self.img_base.set_data(img_base.permute(1,2,0)/3000)
            self.axes[0].set_title(f'Base image : {img_base_currentid[0]}  {img_base_currentid[1]}')
            self.fig.canvas.draw()
        
        if event.key =='left':
            # Update the image base id and idx
            self.img_base_currentid_idx = (self.img_base_currentid_idx-1)%len(self.img_ids)
            img_base_currentid = self.img_ids[self.img_base_currentid_idx]
            self.base_img_id = img_base_currentid

            # retrieve the corresponding image
            img_base_path = os.path.join(self.dataset_path, img_base_currentid[0], img_base_currentid[1], 'hr_res.tif')
            with rasterio.open(img_base_path) as src:
                img_base = torch.from_numpy(src.read())

            # Update the axes[0] and the title
            self.img_base.set_data(img_base.permute(1,2,0)/3000)
            self.axes[0].set_title(f'Base image : {img_base_currentid[0]}  {img_base_currentid[1]}')
            self.fig.canvas.draw()


        if event.key == 'up': # Detail image
            # Update the image base id and idx
            self.img_det_currentid_idx = (self.img_det_currentid_idx+1)%len(self.img_ids)
            img_det_currentid = self.img_ids[self.img_det_currentid_idx]

            # retrieve the corresponding image
            img_det_path = os.path.join(self.dataset_path, img_det_currentid[0], img_det_currentid[1], 'hr_res.tif')
            with rasterio.open(img_det_path) as src:
                img_det = torch.from_numpy(src.read())

            # Update the axes[0] and the title
            self.img_det.set_data(img_det.permute(1,2,0)/3000)
            self.axes[1].set_title(f'Pick a detail from : {img_det_currentid[0]}  {img_det_currentid[1]}')
            self.fig.canvas.draw()

            self.init_overlay()

        if event.key =='down':
            # Update the image base id and idx
            self.img_det_currentid_idx = (self.img_det_currentid_idx-1)%len(self.img_ids)
            img_det_currentid = self.img_ids[self.img_det_currentid_idx]

            # retrieve the corresponding image
            img_det_path = os.path.join(self.dataset_path, img_det_currentid[0], img_det_currentid[1], 'hr_res.tif')
            with rasterio.open(img_det_path) as src:
                img_det = torch.from_numpy(src.read())

            # Update the axes[0] and the title
            self.img_det.set_data(img_det.permute(1,2,0)/3000)
            self.axes[1].set_title(f'Pick a detail from : {img_det_currentid[0]}  {img_det_currentid[1]}')
            self.fig.canvas.draw()

            # When pressing up/down, reinitialize the overlay
            self.init_overlay()

        # Then make an interfave to select a ROI
        if event.key == '0' and len(self.verts) >= 3:
            # Complete current polygon
            ax = self.axes[1]
            poly = Polygon(self.verts, closed=True, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(poly)
            self.polygons_list.append(self.verts.copy())
            self.verts = []
            for line in self.lines:
                line.set_data([], [])
            self.fig.canvas.draw_idle()

            # Get bounding box of polygons + the center point (in img_det),
            #   Apply polygon_mask on the img_det
            #   The preview will be that detail,

            mask_ROI = 1-apply_polygon_mask(self.img_det.get_array(),self.polygons_list)
            xs, ys, bands = np.where(mask_ROI == 1)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            x_det_center = (xmin + xmax)//2
            y_det_center = (ymin + ymax)//2

            #patch_xmin = max(x_det_center-64,0)
            #patch_xmax = min(x_det_center+ 64, self.img_det.get_array().shape[0])

            #patch_ymin = max(y_det_center-64,0)
            #patch_ymax = min(y_det_center+ 64, self.img_det.get_array().shape[1])

            detail_overlay = (mask_ROI* self.img_det.get_array())[xmin:xmax, ymin:ymax, :]

            self.update_overlay_artist(detail_overlay)



        elif event.key == 'd':
            # Remove the last polygon from the list
            removed_polygon = self.polygons_list.pop()
     
            # Remove the display of the removed polygon on each axis
            ax = self.axes[1]
            for patch in ax.patches:
                if isinstance(patch, Polygon) and np.array_equal(patch.get_xy(), removed_polygon + [removed_polygon[0]]):
                    patch.remove()
            self.fig.canvas.draw_idle()
        
        elif event.key == 'c':
            # Cancel current polygon
            print("Current polygon canceled.")
            self.verts = []
            for line in self.lines:
                line.set_data([], [])
            self.fig.canvas.draw_idle()

        elif event.key == 'y':
            # Add 1 point to the current polygon
            self.verts.append((event.xdata, event.ydata))
            x, y = zip(*self.verts)
            for line in self.lines:
                line.set_data(x, y)
            self.fig.canvas.draw_idle()

        if event.key == 'enter':
            arr = self.get_overlay_array()
            self.new_detail = arr
            plt.close()
            

    
    
        # When pressing enter : 
        #   Get bounding box of polygons + the center point (in img_det), 
        #   Other operations to paste the detail with p_null
    

        #self.update_extent()

    def on_click(self, event):
        if event.inaxes != self.axes[0] or event.button != 3:
            return
        # move overlay to center at click
        self.center_x = event.xdata
        self.center_y = event.ydata
        self.update_extent()
        
    def update_extent(self):
        x0 = self.center_x - self.w / 2
        y0 = self.center_y - self.h / 2
        self.overlay_artist.set_extent([x0, x0 + self.w, y0, y0 + self.h])
        self.fig.canvas.draw_idle()
    
class DistanceMetric(ABC):
    """
    Abstract base class for computing distance between two tensors.

    Attributes:
        method: Distance computation method ("pixel", "patch", or "image").
        patch_size: Patch size for patch-based methods.
        x: First input tensor.
        y: Second input tensor.
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
        """
        Splits a tensor into square patches.

        Args:
            tensor: Input tensor (C, H, W).
            patch_size: Size of each patch.

        Returns:
            Tensor of patches (n_patches, n_patches, C, patch_size, patch_size).

        Raises:
            ValueError: If tensor is not square.
        """
        if tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("The tensor must be square.")

        xdim = tensor.shape[1]
        ydim = tensor.shape[2]

        minimages_x = int(torch.ceil(torch.tensor(xdim / patch_size)))
        minimages_y = int(torch.ceil(torch.tensor(ydim / patch_size)))

        pad_x_01 = int((minimages_x * patch_size - xdim) // 2)
        pad_x_02 = int((minimages_x * patch_size - xdim) - pad_x_01)

        pad_y_01 = int((minimages_y * patch_size - ydim) // 2)
        pad_y_02 = int((minimages_y * patch_size - ydim) - pad_y_01)

        padded_tensor = torch.nn.functional.pad(
            tensor, (pad_x_01, pad_x_02, pad_y_01, pad_y_02)
        )

        patches = padded_tensor.unfold(1, patch_size, patch_size).unfold(
            2, patch_size, patch_size
        )

        patches = patches.permute(1, 2, 0, 3, 4)

        return patches

    @abstractmethod
    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def compute_image(self) -> torch.Tensor:
        """Computes the distance at the image level."""
        return self._compute_image(self.x, self.y)

    def compute_patch(self) -> torch.Tensor:
        """Computes the distance at the patch level."""
        x_batched = self.do_square(self.x, self.patch_size)
        y_batched = self.do_square(self.y, self.patch_size)

        metric_result = torch.zeros(x_batched.shape[:2])
        xrange, yrange = x_batched.shape[0:2]
        for x_index in range(xrange):
            for y_index in range(yrange):
                x_batch = x_batched[x_index, y_index]
                y_batch = y_batched[x_index, y_index]
                metric_result[x_index, y_index] = self._compute_image(x_batch, y_batch)

        metric_result = torch.nn.functional.interpolate(
            metric_result[None, None], size=self.x.shape[-2:], mode="nearest"
        ).squeeze()

        return metric_result

    def compute_pixel(self) -> torch.Tensor:
        """Computes the distance at the pixel level."""
        return self._compute_pixel(self.x, self.y)

    def compute(self) -> torch.Tensor:
        """Computes the distance according to the specified method."""
        if self.method == "pixel":
            return self.compute_pixel()
        elif self.method == "image":
            return self.compute_image()
        elif self.method == "patch":
            return self.compute_patch()
        else:
            raise ValueError("Invalid method.")

class L1(DistanceMetric):
    """L1 distance between two tensors."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y))

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean(torch.abs(x - y), axis=0)

class L2(DistanceMetric):
    """L2 distance between two tensors."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nanmean((x - y) ** 2, axis=0)

class LP(DistanceMetric):
    """Lp distance between two tensors."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p: int,
        method: str = "image",
        patch_size: int = 32,
    ):
        self.p = p
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.nanmean(torch.abs(x - y) ** self.p)) ** (1 / self.p)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.nanmean(torch.abs(x - y) ** self.p)) ** (1 / self.p)

class Struct(DistanceMetric):
    """Structural similarity term (SSIM without constants)."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        method: str = "image",
        patch_size: int = 32,
    ):
        super().__init__(x=x, y=y, method=method, patch_size=patch_size)

    def _compute_image(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_center = x - torch.mean(x)
        y_center = y - torch.mean(y)

        dot_product = (x_center * y_center).squeeze().sum()
        preds_norm = x_center.squeeze().norm()
        target_norm = y_center.squeeze().norm()
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.rad2deg(sam_score)

    def _compute_pixel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_center = x - torch.mean(x)
        y_center = y - torch.mean(y)
        dot_product = (x_center * y_center).sum(dim=0)
        preds_norm = x_center.norm(dim=0)
        target_norm = y_center.norm(dim=0)
        sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
        return torch.rad2deg(sam_score)

def depatchify(patched_image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Converts a patched image back to a grid of patch values.

    Args:
        patched_image: Patched image array.
        patch_size: Size of each patch.

    Returns:
        Depatchified image as a numpy array.
    """
    h, w = patched_image.shape
    x_coords = list(range(0, h, patch_size))
    y_coords = list(range(0, w, patch_size))
    nr = len(x_coords)
    nc = len(y_coords)
    depatched_image = np.zeros((nr, nc))
    for i in range(nr):
        for j in range(nc):
            depatched_image[i, j] = patched_image[x_coords[i], y_coords[j]]
    return depatched_image

def repatchify(depatched_image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Converts a depatchified image back to the original image shape.

    Args:
        depatched_image: Depatchified image array.
        patch_size: Size of each patch.

    Returns:
        Patched image as a numpy array.
    """
    if len(depatched_image.shape) == 2:
        nr, nc = depatched_image.shape
        patched_image = np.zeros((nr * patch_size, nc * patch_size))
        for i in range(nr):
            for j in range(nc):
                imin = i * patch_size
                imax = imin + patch_size

                jmin = j * patch_size
                jmax = jmin + patch_size
                patched_image[imin:imax, jmin:jmax] = depatched_image[i, j]

        return patched_image

    elif len(depatched_image.shape) == 3:
        b, nr, nc = depatched_image.shape
        patched_image = np.zeros((b, nr * patch_size, nc * patch_size))

        for i in range(nr):
            for j in range(nc):
                imin = i * patch_size
                imax = imin + patch_size

                jmin = j * patch_size
                jmax = jmin + patch_size
                for k in range(b):
                    patched_image[k, imin:imax, jmin:jmax] = depatched_image[k, i, j]
        return patched_image

def get_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    agg_method: str,
    patch_size: int = 32,
    scale: int = 4,
    device: Union[str, torch.device] = "cpu",
    rgb_bands: Optional[List[int]] = [0, 1, 2],
    p_q: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Estimates the distance between two tensors.

    Args:
        x: First input tensor (C, H, W).
        y: Second input tensor (C, H, W).
        method: Distance method ("l1", "l2", "struct", "lp", "aggpatch").
        agg_method: Aggregation method ("pixel", "image", "patch").
        patch_size: Patch size for patch-based methods.
        scale: Super-resolution scale.
        device: Device to use for computation.
        rgb_bands: Bands to use for RGB images.
        p_q: Tuple (p, q) for aggpatch method.

    Returns:
        Distance value as a torch.Tensor.

    Raises:
        ValueError: If method or q is not supported.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("The number of channels in x and y must be the same.")
    if x.shape[1] != y.shape[1]:
        raise ValueError("The height of x and y must be the same.")

    if method == 'aggpatch':
        if p_q is None:
            raise ValueError('p and q not specified')
        p, q = p_q
        if q == 1:
            distance_fn = L1(x=x, y=y, method='patch', patch_size=patch_size)
        elif q == 2:
            distance_fn = L2(x=x, y=y, method='patch', patch_size=patch_size)
        else:
            raise ValueError('This q order is not supported for aggpatch')

        patch_distances = torch.tensor(depatchify(distance_fn.compute(), patch_size=patch_size) ** (1 / q))

        if p == np.inf:
            return torch.max(patch_distances)

        return torch.norm(patch_distances, p=p)

    if method == "l1":
        distance_fn = L1(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "l2":
        distance_fn = L2(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method == "struct":
        distance_fn = Struct(x=x, y=y, method=agg_method, patch_size=patch_size)
    elif method[:2] == "lp":
        p = int(method[2:])
        distance_fn = LP(x=x, y=y, p=p, method=agg_method, patch_size=patch_size)
    else:
        raise ValueError("No valid distance method.")

    return distance_fn.compute()

def run_opensr_model(
    model: opensr_model,
    lr: np.ndarray,
    hr: np.ndarray,
    device: Union[str, torch.device] = "cpu"
) -> dict:
    """
    Runs the OpenSR model on low-resolution and high-resolution images.

    Args:
        model: OpenSR model instance.
        lr: Low-resolution image as a numpy array.
        hr: High-resolution image as a numpy array.
        device: Device to use for computation.

    Returns:
        Dictionary containing 'lr', 'sr', and 'hr' as numpy arrays.
    """
    if lr.shape[0] == 12:
        lr_img = torch.tensor(lr[[3, 2, 1, 7]] / 10000).to(device).float()
    else:
        lr_img = torch.tensor(lr / 10000).to(device).float()
    hr_img = hr[0:3]

    if lr_img.shape[1] == 121:
        lr_img = torch.nn.functional.pad(
            lr_img[None],
            pad=(3, 4, 3, 4),
            mode='reflect'
        ).squeeze()

        with torch.no_grad():
            sr_img = model(lr_img[None]).squeeze()

        lr_img = lr_img[:, 3:-4, 3:-4]
        sr_img = sr_img[:, 3*4:-4*4, 3*4:-4*4]
    else:
        with torch.no_grad():
            sr_img = model(lr_img[None]).squeeze()

    lr_img = (lr_img.cpu().numpy()[0:3] * 10000).astype(np.uint16)
    sr_img = (sr_img.cpu().numpy()[0:3] * 10000).astype(np.uint16)
    hr_img = hr_img

    return {
        "lr": lr_img,
        "sr": sr_img,
        "hr": hr_img
    }

def apply_square_op_small(Op_Mat: np.ndarray, img: Union[np.ndarray, torch.Tensor], out_2Dshape: Tuple[int, int]) -> torch.Tensor:
    """
    Applies a square operator matrix to each channel of an image.

    Args:
        Op_Mat: Operator matrix.
        img: Image array (C, H, W).
        out_2Dshape: Output 2D shape.

    Returns:
        Transformed image as a torch.Tensor.
    """
    matlist = []
    for i in range(img.shape[0]):
        matlist.append((Op_Mat @ np.asarray(img[i]).flatten()).reshape(out_2Dshape))
    return torch.tensor(np.stack(matlist))

def find_lr_data_idx(lr_res: torch.Tensor, subdataset: str) -> int:
    """
    Finds the index of the closest LR data in the specified subdataset.

    Args:
        lr_res: Reference LR tensor.
        subdataset: Subdataset name.

    Returns:
        Index of the closest LR data.
    """
    n_img_subds = len(os.listdir(f'/localhome/iaga_dv/Dokumente/sat_data/cross_processed/{subdataset}'))
    deltas = []
    for i in range(n_img_subds):
        lr_data_path = f'/localhome/iaga_dv/Dokumente/sat_data/cross_processed/{subdataset}/{i}/lr_data.tif'
        with rasterio.open(lr_data_path) as src:
            lr_data = torch.from_numpy(src.read())[[3, 2, 1]]

        deltas.append(torch.norm(lr_data - lr_res))

    i_corresp = np.argmin(np.array(deltas))
    return i_corresp

def apply_square_op_full(Op_mat: np.ndarray, img: torch.Tensor, out_2D_shape_op: Tuple[int, int], border: int = 4) -> torch.Tensor:
    """
    Applies a sparse square operator to an image in big patches.

    Args:
        Op_mat: Operator matrix.
        img: Image tensor (C, H, W).
        out_2D_shape_op: Output 2D shape for operator.
        border: Border size.

    Returns:
        Transformed image as a torch.Tensor.
    """
    c, h, w = img.shape
    b, a = out_2D_shape_op
    n_y = (h) // (b - 2 * border)
    n_x = (w) // (a - 2 * border)

    OP_img = torch.zeros_like(img)
    for i in range(n_y):
        for j in range(n_x):
            imin = i * (b - 2 * border)
            imax = imin + b

            jmin = j * (a - 2 * border)
            jmax = jmin + a

            if jmax <= w and imax <= h:
                if i == 0 and j == 0:
                    Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:, imin:imax-border, jmin:jmax-border] = Apatch[:, :-border, :-border]
                elif i == 0 and j > 0:
                    Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:, imin:imax-border, jmin+border:jmax-border] = Apatch[:, :-border, border:-border]
                elif i > 0 and j == 0:
                    Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:, imin+border:imax-border, jmin:jmax-border] = Apatch[:, border:-border, :-border]

                else:
                    Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                    OP_img[:, imin+border:imax-border, jmin+border:jmax-border] = Apatch[:, border:-border, border:-border]

    # Do it for the last row, col
    for i in range(n_y):
        imin = i * (b - 2 * border)
        imax = imin + b

        jmin = h - a
        jmax = h

        if imax <= h:
            if i == 0:
                Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:, imin:imax-border, jmin+border:jmax] = Apatch[:, :-border, border:]
            else:
                Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:, imin+border:imax-border, jmin+border:jmax] = Apatch[:, border:-border, border:]

    for j in range(n_x):
        jmin = j * (a - 2 * border)
        jmax = jmin + a

        imin = w - b
        imax = w

        if jmax <= w:
            if j == 0:
                Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:, imin+border:imax, jmin:jmax-border] = Apatch[:, border:, :-border]
            else:
                Apatch = apply_square_op_small(Op_mat, img[:, imin:imax, jmin:jmax], out_2Dshape=out_2D_shape_op)
                OP_img[:, imin+border:imax, jmin+border:jmax-border] = Apatch[:, border:, border:-border]

    Apatch = apply_square_op_small(Op_mat, img[:, w-b:w, h-a:h], out_2Dshape=out_2D_shape_op)
    OP_img[:, w-b+border:w, h-a+border:h] = Apatch[:, border:, border:]
    return OP_img

def save_into_tiff(bands: np.ndarray, out_path: str) -> None:
    """
    Saves a multi-band image as a TIFF file.

    Args:
        bands: Image bands (C, H, W).
        out_path: Output file path.
    """
    if isinstance(bands, torch.Tensor):
        bands = np.array(bands)
    with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=bands.shape[1],
                width=bands.shape[2],
                count=bands.shape[0],
                dtype='float32',
                crs='+proj=latlong',
                transform=from_origin(0, 0, 10, 10)
                ) as dst:
                for i in range(bands.shape[0]):
                        dst.write(bands[i, :, :].astype(np.float32), i + 1)

def get_patches_from_S2(img: torch.Tensor, patchsize: int, border: int) -> torch.Tensor:
    """
    Extracts patches from an image with a given patch size and border.

    Args:
        img: Image tensor (C, H, W).
        patchsize: Patch size.
        border: Border size.

    Returns:
        Stacked patches as a torch.Tensor.
    """
    h, w = img.shape[1:]
    n_patches_y = (h - border - patchsize) // patchsize
    n_patches_x = (w - border - patchsize) // patchsize
    all_patches = []
    for i in range(n_patches_y):
        imin = border + i * patchsize
        imax = imin + patchsize
        for j in range(n_patches_x):
            jmin = border + j * patchsize
            jmax = jmin + patchsize
            all_patches.append(img[:, imin:imax, jmin:jmax])

    all_patches = torch.stack(all_patches)
    return all_patches

def get_feasible_info(distsXX, feasible_appartenance):
    """
    Computes information for each feasible set F_y:
    - Diameter of F_y (maximum pairwise distance within F_y),
    - Indices of elements corresponding to the diameter,
    - Cardinality of F_y (number of elements in the feasible set).

    Args:
        distsXX: Pairwise distance matrix between target samples.
        feasible_appartenance: Feasible appartenance matrix.

    Returns:
        List of (diam_Fy, [i, j], cardinality) for each target sample.
    """
    def get_info(y_idx, fa, dXX):
        valid_idx = fa[:, y_idx].nonzero()[0]
        subdistXX = dXX[valid_idx, :][:, valid_idx]
        subdistXX = subdistXX.toarray()

        if subdistXX.size == 0:
            return 0, (None, None), 0

        diam_Fy = np.nanmax(subdistXX)
        flat_index = np.nanargmax(subdistXX)
        row, col = np.unravel_index(flat_index, subdistXX.shape)

        i = valid_idx[row]
        j = valid_idx[col]

        return float(diam_Fy), [int(i), int(j)], int(subdistXX.shape[0])
    n, p = feasible_appartenance.shape

    return list(Parallel(n_jobs=-1, backend='threading')(delayed(get_info)(y_idx, feasible_appartenance, distsXX) for y_idx in tqdm(range(p))))

def build_S2_patched_dataset_DSHR(
    patchsize_X: int,
    img_dset_folder: str,
    subdatasets: List[str],
    out_dsfolder: str,
    labels: Tuple[str, str] = ('hr_res', 'lr_res'),
    border_X: int = 0,
    SR_factor: int = 4
) -> None:
    """
    Builds a patched dataset for S2 images with downsampled high-resolution images as LR images.

    Args:
        patchsize_X: Patch size for HR images.
        img_dset_folder: Input dataset folder.
        subdatasets: List of subdataset names.
        out_dsfolder: Output folder for patches.
        labels: Labels for HR and LR images.
        border_X: Border size for HR images.
        SR_factor: Super-resolution factor.
    """
    border_Y = border_X // SR_factor
    patchsize_Y = patchsize_X // SR_factor

    hr_label = labels[0]
    lr_label = labels[1]

    for subds in subdatasets:
        print()
        print(f'Subdataset : {subds}')

        data_folder = os.path.join(img_dset_folder, subds)
        img_folders = os.listdir(data_folder)
        bar = [x for x in img_folders if 'json' not in x]

        for idxstr in tqdm(bar):
            img_folder = os.path.join(data_folder, idxstr)
            hr_path = f'{img_folder}/{hr_label}.tif'

            with rasterio.open(hr_path) as hr_src:
                hr_img = hr_src.read()
            hr_img = torch.from_numpy(hr_img)

            lr_img = apply_upsampling(torch.tensor(hr_img), scale=SR_factor)

            patched_lr = get_patches_from_S2(lr_img, patchsize=patchsize_Y, border=border_Y)
            patched_hr = get_patches_from_S2(hr_img, patchsize_X, border_X)

            m = patched_lr.shape[0]
            for i in range(m):
                save_into_tiff(bands=np.array(patched_lr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{lr_label}.tif'))
                save_into_tiff(bands=np.array(patched_hr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{hr_label}.tif'))

def build_S2_patched_dataset(
    patchsize_X: int,
    img_dset_folder: str,
    subdatasets: List[str],
    out_dsfolder: str,
    labels: Tuple[str, str] = ('hr_data', 'lr_data'),
    border_X: int = 0,
    SR_factor: int = 4
) -> None:
    """
    Builds a patched dataset for S2 images.

    Args:
        patchsize_X: Patch size for HR images.
        img_dset_folder: Input dataset folder.
        subdatasets: List of subdataset names.
        out_dsfolder: Output folder for patches.
        labels: Labels for HR and LR images.
        border_X: Border size for HR images.
        SR_factor: Super-resolution factor.
    """
    border_Y = border_X // SR_factor
    patchsize_Y = patchsize_X // SR_factor

    hr_label = labels[0]
    lr_label = labels[1]

    for subds in subdatasets:
        data_folder = os.path.join(img_dset_folder, subds)
        img_folders = os.listdir(data_folder)
        bar = [x for x in img_folders if 'json' not in x]

        for idxstr in tqdm(bar):
            img_folder = os.path.join(data_folder, idxstr)
            lr_path = f'{img_folder}/{lr_label}.tif'
            hr_path = f'{img_folder}/{hr_label}.tif'

            with rasterio.open(hr_path) as hr_src:
                hr_img = hr_src.read()
            hr_img = torch.from_numpy(hr_img)
            with rasterio.open(lr_path) as lr_src:
                lr_img = lr_src.read()
            lr_img = torch.from_numpy(lr_img)

            patched_lr = get_patches_from_S2(lr_img, patchsize=patchsize_Y, stride=patchsize_Y, border=border_Y)
            patched_hr = get_patches_from_S2(hr_img, patchsize_X, patchsize_X, border_X)

            m = patched_lr.shape[0]
            for i in range(m):
                save_into_tiff(bands=np.array(patched_lr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{lr_label}.tif'))
                save_into_tiff(bands=np.array(patched_hr[i]), out_path=os.path.join(out_dsfolder, f'{subds}_{idxstr}_{i}_{hr_label}.tif'))

def apply_upsampling(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Upsamples a tensor to a lower resolution using bilinear interpolation with antialiasing.

    Args:
        x: Input tensor (B, C, H, W).
        scale: Super-resolution scale.

    Returns:
        Upsampled tensor (B, C, H', W').
    """
    x_ref = torch.nn.functional.interpolate(
        input=x[None], scale_factor=1 / scale, mode="bilinear", antialias=True
    ).squeeze()

    return x_ref

def bilinear_SR(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Upsamples a tensor to a higher resolution using bilinear interpolation with antialiasing.

    Args:
        x: Input tensor (B, C, H, W).
        scale: Super-resolution scale.

    Returns:
        Upsampled tensor (B, C, H', W').
    """
    x_ref = torch.nn.functional.interpolate(
        input=x[None], scale_factor=scale, mode="bilinear", antialias=True
    ).squeeze()

    return x_ref

def bicubic_SR(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Upsamples a tensor to a higher resolution using bicubic interpolation with antialiasing.

    Args:
        x: Input tensor (B, C, H, W).
        scale: Super-resolution scale.

    Returns:
        Upsampled tensor (B, C, H', W').
    """
    x_ref = torch.nn.functional.interpolate(
        input=x[None], scale_factor=scale, mode="bicubic", antialias=True
    ).squeeze()

    return x_ref


class ImgComparator:
    """
    Synchronizes zoom/pan across multiple matplotlib axes.

    Args:
        fig (matplotlib.figure.Figure): Figure object.
        axlist (list, optional): List of axes to synchronize.
    """
    def __init__(self, fig, axlist = None):
        self.canvas = fig.canvas
        if axlist is None:
            self.axlist = fig.axes
        else:
            self.axlist = axlist
        self.cid_zoom = fig.canvas.mpl_connect('motion_notify_event', self.on_zoom)
    def on_zoom(self, event):
        if event.inaxes:
            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()
            for ax in self.axlist:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.canvas.draw_idle()

def rescale_plot(img: torch.Tensor) -> torch.Tensor:
    """
    Rescales an image tensor to [0, 1] for plotting.

    Args:
        img: Image tensor.

    Returns:
        Rescaled image.
    """
    minval = torch.min(img)
    maxval = torch.max(img)
    return (img - minval) / (maxval - minval)

class MatrixOpCalculator:
    """
    Computes and manipulates sparse matrix operators.

    Attributes:
        n_in: Input dimension.
        n_out: Output dimension.
        operator: Operator function.
        num_workers: Number of parallel workers.
        singular_threshold_ratio: Threshold for singular values.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        Operator,
        num_workers: Optional[int] = None,
        singular_threshold_ratio: float = 0.001
    ):
        self.n_in = n_in
        self.n_out = n_out
        self.operator = Operator
        self.num_workers = num_workers
        if num_workers is None:
            self.num_workers = cpu_count()
        self.singular_threshold_ratio = singular_threshold_ratio

    def compute_column(self, i: int):
        """
        Computes a single column of the sparse matrix.

        Args:
            i: Column index.

        Returns:
            Tuple of (row_indices, column_indices, data) for the column.
        """
        A = self.operator
        e_i = np.zeros(self.n_in)
        e_i[i] = 1.0
        col = A(e_i)

        row_idx = np.nonzero(col)[0]
        data = col[row_idx]
        col_idx = np.full_like(row_idx, i)

        return row_idx, col_idx, data

    def build_sparse_matrix_parallel(self):
        """
        Builds the sparse matrix in parallel.

        Returns:
            Sparse matrix in CSC format.
        """
        n_in = self.n_in
        n_out = self.n_out

        with Pool(self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.compute_column, range(n_in)), total=n_in))

        row_indices = []
        col_indices = []
        data = []

        for r, c, d in results:
            row_indices.extend(r)
            col_indices.extend(c)
            data.extend(d)

        A_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(n_out, n_in)).tocsc()
        return A_sparse

    def get_range_space_basis(self, A_sparse, sigma_threshold_ratio: float = 0.001):
        """
        Computes the range space basis of the sparse matrix.

        Args:
            A_sparse: Sparse matrix.
            sigma_threshold_ratio: Threshold for singular values.

        Returns:
            Range space basis as a numpy array.
        """
        p, q = A_sparse.shape
        kmax = int(min(p, q) - 1)
        umat, sing, vt = svds(A_sparse, k=kmax)

        threshold = sigma_threshold_ratio * np.max(sing)
        range_space_basis = vt[sing >= threshold].T
        return range_space_basis

    def make_null_projection_operator(self, range_basis):
        """
        Creates a null projection operator from the range basis.

        Args:
            range_basis: Range space basis.

        Returns:
            Null projection operator as a numpy array.
        """
        n = range_basis.shape[0]
        return np.eye(n) - range_basis.dot(range_basis.T)
