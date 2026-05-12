import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from scipy.ndimage import zoom, gaussian_filter
import h5py
import torch
from tqdm import tqdm
from skimage.filters import threshold_otsu
from scipy.ndimage import label, binary_closing, zoom
import cv2
import fastmri.data.transforms as T
from fastmri.models import Unet

class ImgComparator:
    """
    Synchronizes zoom/pan across multiple matplotlib axes.

    Attributes:
        canvas (matplotlib.backend_bases.FigureCanvasBase): Figure canvas.
        axlist (list): List of axes to synchronize.
        cid_zoom (int): Connection ID for zoom event.
    """
    def __init__(self, fig, axlist=None):
        """
        Args:
            fig (matplotlib.figure.Figure): Figure object.
            axlist (list, optional): List of axes to synchronize. If None, all axes in the figure are used.
        """
        self.canvas = fig.canvas
        if axlist is None:
            self.axlist = fig.axes
        else:
            self.axlist = axlist
        self.cid_zoom = fig.canvas.mpl_connect('motion_notify_event', self.on_zoom)

    def on_zoom(self, event):
        """
        Callback for zoom/pan events. Synchronizes all axes in axlist.
        """
        if event.inaxes:
            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()
            for ax in self.axlist:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.canvas.draw_idle()

def inverse_fft2_shift(kspace):
    """
    Compute the inverse 2D FFT with shift for k-space data.

    Args:
        kspace (np.ndarray): Input k-space data.

    Returns:
        np.ndarray: Image domain data.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))

def dft2(x):
    """
    Compute the 2D FFT with shift for image data.

    Args:
        x (np.ndarray): Input image data.

    Returns:
        np.ndarray: K-space data.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))

def draw_noise(image, sigma, mask):
    """
    Add Gaussian noise to an image, masked by a binary mask.

    Args:
        image (np.ndarray): Input image.
        sigma (float): Standard deviation of noise.
        mask (np.ndarray): Binary mask (1=apply noise, 0=do not apply).

    Returns:
        np.ndarray: Noisy image.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    if np.iscomplexobj(image):
        noise = noise + 1j * np.random.normal(loc=0.0, scale=sigma, size=image.shape)
    noise *= mask
    return image + noise

def generate_mask(accel_rate, n_central, shape_k):
    """
    Generate a subsampling mask for k-space.

    Args:
        accel_rate (float): Acceleration rate.
        n_central (int): Number of central lines to keep.
        shape_k (tuple): Shape of k-space (H, W).

    Returns:
        np.ndarray: Binary subsampling mask.
    """
    h, w = shape_k
    mask = np.zeros(shape_k)
    interval = int((w - n_central) // (h / accel_rate - n_central))

    chosen_bands1 = list(range(0, (w - n_central) // 2, interval))
    chosen_bands2 = list(range((w + n_central) // 2, w, interval))

    for x in chosen_bands1:
        mask[:, x] = 1
    for x in chosen_bands2:
        mask[:, x] = 1

    mask[:, (w - n_central) // 2 : (w + n_central) // 2] = 1
    return mask

class ImgPasting:
    """
    Interactive tool for pasting anomalies onto a base image.

    Attributes:
        anomalies_assoc (dict): Maps anomaly types to lists of slice indices.
        slicelist (list): List of all slice indices.
        anomaly_types (list): List of anomaly types.
        folder_scans (str): Path to folder containing scan images.
        folder_ROI (str): Path to folder containing ROI annotations.
        current_anomaly_idx (int): Index of current anomaly type.
        current_anom_nb (int): Index of current anomaly in type.
        preview_img (np.ndarray): Current preview image.
        H_base (int): Height of base image.
        W_base (int): Width of base image.
        overlay (np.ndarray): Current overlay image.
        fig (matplotlib.figure.Figure): Figure object.
        ax (matplotlib.axes.Axes): Main axes.
        ax_preview (matplotlib.axes.Axes): Preview axes.
        h (int): Height of overlay.
        w (int): Width of overlay.
        center_x (float): X-coordinate of overlay center.
        center_y (float): Y-coordinate of overlay center.
        overlay_artist (matplotlib.image.AxesImage): Overlay image artist.
    """
    def __init__(self, fig, base_shape=(320, 320), anomalies_assoc={'coucou': ['nada']}, slicelist=['nada'], folder_scans='', folder_ROI=''):
        """
        Args:
            fig (matplotlib.figure.Figure): Figure object.
            base_shape (tuple): Shape of the base image (H, W).
            anomalies_assoc (dict): Maps anomaly types to lists of slice indices.
            slicelist (list): List of all slice indices.
            folder_scans (str): Path to folder containing scan images.
            folder_ROI (str): Path to folder containing ROI annotations.
        """
        self.anomalies_assoc = anomalies_assoc
        self.slicelist = slicelist
        self.anomaly_types = list(self.anomalies_assoc.keys())
        self.folder_scans = folder_scans
        self.folder_ROI = folder_ROI

        self.current_anomaly_idx = 0
        self.current_anom_nb = 0
        self.preview_img = None

        self.H_base, self.W_base = base_shape
        self.overlay = np.ones((100, 100))
        self.fig = fig
        self.ax = fig.axes[0]
        self.ax_preview = fig.axes[1]

        x0, y0 = 50, 50
        h, w = self.overlay.shape[:2]
        self.h = h
        self.w = w
        self.center_x = x0 + w / 2
        self.center_y = y0 + h / 2

        self.overlay_artist = self.ax.imshow(
            self.overlay,
            extent=[x0, x0 + w, y0, y0 + h],
            alpha=0.7,
            cmap='gray',
            origin='upper'
        )

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def open_parallel_plot(self, data):
        """
        Open or refresh a separate non-blocking window showing the anomaly.

        Args:
            data (np.ndarray): Image data to display.
        """
        self.ax_preview.clear()
        self.ax_preview.imshow(data, origin='upper')
        self.ax_preview.set_title(f"Anomaly in its original scan")
        self.fig.canvas.draw_idle()

    def update_overlay_artist(self, new_overlay):
        """
        Update the overlay artist with a new overlay image.

        Args:
            new_overlay (np.ndarray): New overlay image.
        """
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
        self.fig.canvas.draw_idle()

    def init_overlay(self, slice_info):
        """
        Initialize overlay from a slice.

        Args:
            slice_info (str): Slice identifier.
        """
        idx = self.slicelist.index(slice_info)
        slice_img = np.load(f'{self.folder_scans}/idx_{idx}.npy')
        self.open_parallel_plot(slice_img)
        self.anomalous_slice = slice_img

        with open(f'{self.folder_ROI}/idx_{idx}.json', 'r') as f:
            polygons = json.load(f)
        if len(polygons) > 0:
            mask_ROI = 1 - apply_polygon_mask(np.ones((320, 320)), [polygons[0]])
        else:
            return np.ones((100, 100))

        xs, ys = np.where(mask_ROI == 1)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        anomaly_img = (mask_ROI * slice_img)[xmin:xmax, ymin:ymax]
        self.update_overlay_artist(anomaly_img)

    def on_click(self, event):
        """
        Callback for mouse click events. Moves overlay to click position.
        """
        if event.inaxes != self.ax or event.button != 3:
            return
        self.center_x = event.xdata
        self.center_y = event.ydata
        self.update_extent()

    def on_key_press(self, event):
        """
        Callback for key press events. Handles overlay resizing, navigation, and saving.
        """
        if event.key == '+':
            self.w *= 1.05
            self.h *= 1.05
        elif event.key == '-':
            self.w *= 0.95
            self.h *= 0.95
        elif event.key == 'enter':
            arr = self.get_overlay_array()
            self.new_detail = arr
            print("Overlay saved as array with shape:", arr.shape)
            plt.close()
        elif event.key == 'up':
            self.current_anomaly_idx = (self.current_anomaly_idx + 1) % len(self.anomaly_types)
            slice = self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]][self.current_anom_nb]
            print(f'Anomaly type: {self.anomaly_types[self.current_anomaly_idx]} \nIdx for that type: {self.current_anom_nb + 1}/{len(self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]])}')
            self.init_overlay(slice)
        elif event.key == 'down':
            self.current_anomaly_idx = (self.current_anomaly_idx - 1) % len(self.anomaly_types)
            slice = self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]][self.current_anom_nb]
            print(f'Anomaly type: {self.anomaly_types[self.current_anomaly_idx]} \nIdx for that type: {self.current_anom_nb + 1}/{len(self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]])}')
            self.init_overlay(slice)
        elif event.key == 'right':
            self.current_anom_nb = (self.current_anom_nb + 1) % len(self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]])
            slice = self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]][self.current_anom_nb]
            print(f'Anomaly type: {self.anomaly_types[self.current_anomaly_idx]} \nIdx for that type: {self.current_anom_nb + 1}/{len(self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]])}')
            self.init_overlay(slice)
        elif event.key == 'left':
            self.current_anom_nb = (self.current_anom_nb - 1) % len(self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]])
            slice = self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]][self.current_anom_nb]
            print(f'Anomaly type: {self.anomaly_types[self.current_anomaly_idx]} \nIdx for that type: {self.current_anom_nb + 1}/{len(self.anomalies_assoc[self.anomaly_types[self.current_anomaly_idx]])}')
            self.init_overlay(slice)

        self.w = max(self.w, 5)
        self.h = max(self.h, 5)
        self.update_extent()

    def update_extent(self):
        """
        Update the extent of the overlay artist.
        """
        x0 = self.center_x - self.w / 2
        y0 = self.center_y - self.h / 2
        self.overlay_artist.set_extent([x0, x0 + self.w, y0, y0 + self.h])
        self.fig.canvas.draw_idle()

    def get_overlay_array(self):
        """
        Return a numpy array of shape (H_base, W_base) with overlay positioned, rest zeros.

        Returns:
            np.ndarray: Array with overlay positioned.
        """
        arr = np.zeros((self.H_base, self.W_base), dtype=self.overlay.dtype)
        h_cur = int(round(self.h))
        w_cur = int(round(self.w))
        overlay_resized = zoom(self.overlay, (h_cur / self.overlay.shape[0], w_cur / self.overlay.shape[1]), order=1)
        x0 = int(round(self.center_x - w_cur / 2))
        y0 = int(round(self.center_y - h_cur / 2))
        x1 = x0 + w_cur
        y1 = y0 + h_cur

        x0_clamped = max(0, x0)
        y0_clamped = max(0, y0)
        x1_clamped = min(self.W_base, x1)
        y1_clamped = min(self.H_base, y1)

        overlay_x0 = x0_clamped - x0
        overlay_y0 = y0_clamped - y0
        overlay_x1 = overlay_x0 + (x1_clamped - x0_clamped)
        overlay_y1 = overlay_y0 + (y1_clamped - y0_clamped)

        arr[320 - y1_clamped:320 - y0_clamped, x0_clamped:x1_clamped] = overlay_resized[overlay_y0:overlay_y1, overlay_x0:overlay_x1]
        return arr

class ManualPolygonDrawer:
    """
    Interactive tool for drawing polygons on images.

    Attributes:
        ax_poly_list (list): List of axes to draw on.
        canvas (matplotlib.backend_bases.FigureCanvasBase): Figure canvas.
        verts (list): List of polygon vertices.
        lines (list): List of line artists for polygon edges.
        cid_click (int): Connection ID for click event.
        cid_key (int): Connection ID for key press event.
        cid_zoom (int): Connection ID for zoom event.
        polygons_dict (dict): Dictionary of polygons for each group.
        outpath (str): Path to save polygons.
        group_idx (int): Current group index.
    """
    def __init__(self, fig, group_idx, outpath):
        """
        Args:
            fig (matplotlib.figure.Figure): Figure object.
            group_idx (int): Current group index.
            outpath (str): Path to save polygons.
        """
        self.ax_poly_list = fig.axes
        self.canvas = fig.canvas
        self.verts = []
        self.lines = [ax.plot([], [], marker='o', color='cyan', linestyle='-')[0] for ax in self.ax_poly_list]
        self.cid_click = self.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_zoom = fig.canvas.mpl_connect('motion_notify_event', self.on_zoom)
        if os.path.exists(outpath):
            with open(outpath, 'r') as f:
                self.polygons_dict = json.load(f)
            if group_idx not in self.polygons_dict:
                self.polygons_dict[group_idx] = []
            for ax in self.ax_poly_list:
                for poly in self.polygons_dict[group_idx]:
                    poly = np.array(poly)
                    closed_poly = np.concatenate([poly, [poly[0]]], axis=0)
                    poly_patch = Polygon(closed_poly, closed=True, fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(poly_patch)
        else:
            self.polygons_dict = {i: [] for i in range(5)}
        self.outpath = outpath
        self.group_idx = group_idx

    def on_click(self, event):
        """
        Callback for mouse click events. Adds a vertex to the current polygon.
        """
        if event.inaxes in self.ax_poly_list:
            self.verts.append((event.xdata, event.ydata))
            x, y = zip(*self.verts)
            for line in self.lines:
                line.set_data(x, y)
            self.canvas.draw_idle()

    def on_key(self, event):
        """
        Callback for key press events. Handles polygon completion, deletion, and saving.
        """
        if event.key == '0' and len(self.verts) >= 3:
            for ax in self.ax_poly_list:
                poly = Polygon(self.verts, closed=True, fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(poly)
            self.polygons_dict[self.group_idx].append(self.verts.copy())
            self.verts = []
            for line in self.lines:
                line.set_data([], [])
            self.canvas.draw_idle()
        elif event.key == 'd':
            if self.polygons_dict[self.group_idx]:
                removed_polygon = self.polygons_dict[self.group_idx].pop()
                for ax in self.ax_poly_list:
                    for patch in ax.patches:
                        if isinstance(patch, Polygon) and np.array_equal(patch.get_xy(), removed_polygon + [removed_polygon[0]]):
                            patch.remove()
                self.canvas.draw_idle()
        elif event.key == 'c':
            print("Current polygon canceled.")
            self.verts = []
            for line in self.lines:
                line.set_data([], [])
            self.canvas.draw_idle()
        elif event.key == 'enter':
            n_polygons = len(self.polygons_dict[self.group_idx])
            print(f"Finished. {n_polygons} polygons selected.")
            self.canvas.mpl_disconnect(self.cid_click)
            self.canvas.mpl_disconnect(self.cid_key)
            self.canvas.mpl_disconnect(self.cid_zoom)
            with open(self.outpath, 'w') as f:
                json.dump(self.polygons_dict, f)
            plt.close()
        elif event.key == 'y':
            if event.inaxes in self.ax_poly_list:
                self.verts.append((event.xdata, event.ydata))
                x, y = zip(*self.verts)
                for line in self.lines:
                    line.set_data(x, y)
                self.canvas.draw_idle()

    def on_zoom(self, event):
        """
        Callback for zoom events. Synchronizes all axes in ax_poly_list.
        """
        if event.inaxes:
            xlim = event.inaxes.get_xlim()
            ylim = event.inaxes.get_ylim()
            for ax in self.ax_poly_list:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            self.canvas.draw_idle()

def apply_polygon_mask(img, polygons):
    """
    Apply a polygon mask to an image.

    Args:
        img (np.ndarray): Input image.
        polygons (list): List of polygons.

    Returns:
        np.ndarray: Masked image.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for poly in polygons:
        path = Path(poly)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
        inside = path.contains_points(coords).reshape(h, w)
        mask |= inside
    img_masked = img.copy()
    if img.ndim == 2:
        img_masked[mask] = 0
    else:
        img_masked[mask] = [0] * img.shape[2]
    return img_masked

def alpha_blending(img_origin, mask_ROI, detail, sigma_blend):
    """
    Blend an image with a detail using a mask.

    Args:
        img_origin (np.ndarray): Original image.
        mask_ROI (np.ndarray): ROI mask.
        detail (np.ndarray): Detail image.
        sigma_blend (float): Sigma for Gaussian blur.

    Returns:
        np.ndarray: Blended image.
    """
    detail += img_origin
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask_ROI, kernel, iterations=1).astype(float)
    blurred_mask = gaussian_filter(eroded_mask, sigma=sigma_blend)
    res_noblend = np.copy(img_origin)
    res_noblend[mask_ROI == 1] = detail[mask_ROI == 1]
    res_blended = blurred_mask * res_noblend + (1 - blurred_mask) * img_origin
    return res_blended

def kernel_draw(GT_img, lim1_img, mask_ROI, P_null, sigma_blend=1, scale=1):
    """
    Draw a kernel projection and blend with the original image.

    Args:
        GT_img (np.ndarray): Ground truth image.
        lim1_img (np.ndarray): Limit image.
        mask_ROI (np.ndarray): ROI mask.
        P_null (callable): Null space projection function.
        sigma_blend (float): Sigma for Gaussian blur.
        scale (float): Scaling factor.

    Returns:
        np.ndarray: Blended image.
    """
    detail = mask_ROI * (scale * lim1_img - GT_img)
    delta_proj = P_null(detail)
    return alpha_blending(GT_img, mask_ROI, delta_proj, sigma_blend=sigma_blend)

def kernel_projection(x, k_mask, max_norm):
    """
    Project an image into the null space of a k-space mask.

    Args:
        x (np.ndarray): Input image.
        k_mask (np.ndarray): K-space mask.
        max_norm (float): Maximum norm.

    Returns:
        np.ndarray: Projected image.
    """
    k_full = dft2(x)
    mask_complementary = 1 - k_mask
    k_mes_norm = np.linalg.norm(k_full * k_mask, ord=2)
    mask_tot = mask_complementary + (max_norm / k_mes_norm) * k_mask
    return inverse_fft2_shift(mask_tot * k_full)

def transform_batch(k_mes_batch, mask2D):
    """
    Transform a batch of k-space images using UnetDataTransform.

    Args:
        k_mes_batch (np.ndarray): Batch of k-space images.
        mask2D (np.ndarray): 2D mask.

    Returns:
        tuple: (transformed images, means, stds)
    """
    n_inputs = k_mes_batch.shape[0]
    transform = T.UnetDataTransform(which_challenge='singlecoil')
    k_img_all = []
    k_mean_all = []
    k_std_all = []
    for i in range(n_inputs):
        k_trans = transform(k_mes_batch[i], mask2D, np.zeros((320, 320)), {'recon_size': k_mes_batch[i].shape}, 'hellooow', 2)
        k_img, k_mean, k_std = k_trans.image.to(torch.float32), k_trans.mean, k_trans.std
        k_img_all.append(k_img)
        k_mean_all.append(k_mean)
        k_std_all.append(k_std)
    k_img_all = torch.stack(k_img_all, dim=0)
    k_mean_all = torch.tensor(k_mean_all)
    k_std_all = torch.tensor(k_std_all)
    return k_img_all, k_mean_all, k_std_all

def predict_batch(model, k_img_batch, means, stds, device):
    """
    Predict a batch of images using a model.

    Args:
        model (torch.nn.Module): Model for prediction.
        k_img_batch (torch.Tensor): Batch of images.
        means (torch.Tensor): Means for normalization.
        stds (torch.Tensor): Standard deviations for normalization.
        device (str): Device to run the model on.

    Returns:
        np.ndarray: Predicted images.
    """
    pred = model(k_img_batch.to(device).unsqueeze(1)).squeeze(1).cpu()
    means = means.unsqueeze(-1).unsqueeze(-1)
    stds = stds.unsqueeze(-1).unsqueeze(-1)
    pred = (pred * stds + means).detach().cpu().numpy()
    return pred

def load_model(device):
    """
    Load a pretrained Unet model.

    Args:
        device (str): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    challenge = 'unet_brain_mc'
    MODEL_FNAMES = {
        "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
        "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
        "unet_brain_mc": "brain_leaderboard_state_dict.pt",
    }
    model_folder = '/localhome/iaga_dv/Dokumente/fastMRI/fastmri_examples/unet'
    state_dict_file = f'{model_folder}/{MODEL_FNAMES[challenge]}'
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    model.load_state_dict(torch.load(state_dict_file, map_location=device))
    model = model.to(device)
    model = model.eval()
    return model

def process_image_V2(k_mes, new_shape_x, margin_ratio=1.5):
    """
    Process a batch of k-space images.

    Args:
        k_mes (np.ndarray): Batch of k-space images.
        new_shape_x (tuple): New shape for output.
        margin_ratio (float): Margin ratio for ROI.

    Returns:
        tuple: (processed k-space, processed images)
    """
    n_slices = k_mes.shape[0]
    k_mes_proc = []
    x_rec_proc = []
    for i in range(n_slices):
        kmes_slice, xrecslice = process_slice_V2(k_mes[i], new_shape_x, margin_ratio)
        k_mes_proc.append(kmes_slice)
        x_rec_proc.append(xrecslice)
    k_mes_proc = np.stack(k_mes_proc, axis=0)
    x_rec_proc = np.stack(x_rec_proc)
    return k_mes_proc, x_rec_proc

def lowfreq_reconstruction_2D(kspace, lf=0.08):
    """
    Reconstruct low-frequency components from k-space.

    Args:
        kspace (np.ndarray): K-space data.
        lf (float): Low-frequency fraction.

    Returns:
        np.ndarray: Low-frequency image.
    """
    H, W = kspace.shape[-2:]
    h0 = int(H * lf / 2)
    w0 = int(W * lf / 2)
    k_low = np.zeros_like(kspace)
    k_low[H // 2 - h0:H // 2 + h0, W // 2 - w0:W // 2 + w0] = kspace[H // 2 - h0:H // 2 + h0, W // 2 - w0:W // 2 + w0]
    img_low = inverse_fft2_shift(k_low)
    img_low_mag = np.abs(img_low)
    return img_low_mag

def lowfreq_reconstruction(kspace, lf=0.08):
    """
    Reconstruct low-frequency components from multi-coil k-space.

    Args:
        kspace (np.ndarray): Multi-coil k-space data.
        lf (float): Low-frequency fraction.

    Returns:
        np.ndarray: Low-frequency image.
    """
    H, W = kspace.shape[-2:]
    h0 = int(H * lf / 2)
    w0 = int(W * lf / 2)
    k_low = np.zeros_like(kspace)
    k_low[:, H // 2 - h0:H // 2 + h0, W // 2 - w0:W // 2 + w0] = kspace[:, H // 2 - h0:H // 2 + h0, W // 2 - w0:W // 2 + w0]
    img_low = inverse_fft2_shift(k_low)
    img_low_mag = np.sqrt(np.sum(np.abs(img_low) ** 2, axis=0))
    return img_low_mag

def envelope_mask(env):
    """
    Create a binary mask from an envelope image using Otsu's thresholding.

    Args:
        env (np.ndarray): Envelope image.

    Returns:
        np.ndarray: Binary mask.
    """
    thr = threshold_otsu(env)
    mask = env > thr
    return mask

def process_slice_V2(k_mes, new_shape_x, margin_ratio=1.15):
    """
    Process a single k-space slice.

    Args:
        k_mes (np.ndarray): K-space slice.
        new_shape_x (tuple): New shape for output.
        margin_ratio (float): Margin ratio for ROI.

    Returns:
        tuple: (processed k-space, processed image)
    """
    x_rec = inverse_fft2_shift(k_mes)
    env = lowfreq_reconstruction(k_mes, lf=0.08)
    mask = envelope_mask(env)
    mask = largest_component(mask)
    box = roi_box(mask, pad_ratio=margin_ratio - 1)
    x_rec_out = crop_and_resize(x_rec, box, new_shape_x)
    k_out = dft2(x_rec_out)
    return k_out, x_rec_out

def largest_component(mask):
    """
    Find the largest connected component in a binary mask.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Mask of the largest component.
    """
    mask = binary_closing(mask, iterations=3)
    lab, num = label(mask)
    if num == 0:
        return mask
    sizes = [(lab == i).sum() for i in range(1, num + 1)]
    i_max = 1 + np.argmax(sizes)
    return (lab == i_max)

def roi_box(mask, pad_ratio=0.15):
    """
    Compute a bounding box for a binary mask with padding.

    Args:
        mask (np.ndarray): Binary mask.
        pad_ratio (float): Padding ratio.

    Returns:
        tuple: (x0, x1, y0, y1) bounding box coordinates.
    """
    xs, ys = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h = x1 - x0
    w = y1 - y0
    H, W = mask.shape
    x0 = max(0, int(x0 - pad_ratio * h))
    x1 = min(H, int(x1 + pad_ratio * h))
    y0 = max(0, int(y0 - pad_ratio * w))
    y1 = min(W, int(y1 + pad_ratio * w))
    return x0, x1, y0, y1

def crop_and_resize(x_rec, box, out_shape):
    """
    Crop and resize an image.

    Args:
        x_rec (np.ndarray): Input image.
        box (tuple): Bounding box coordinates (x0, x1, y0, y1).
        out_shape (tuple): Output shape.

    Returns:
        np.ndarray: Cropped and resized image.
    """
    x0, x1, y0, y1 = box
    crop = x_rec[:, x0:x1, y0:y1]
    _, Hc, Wc = crop.shape
    zoom_factors = (1, out_shape[0] / Hc, out_shape[1] / Wc)
    return zoom(crop, zoom_factors, order=1)

def segment_brain(x_img):
    """
    Segment the brain region from an image.

    Args:
        x_img (np.ndarray): Input image.

    Returns:
        np.ndarray: Binary brain mask.
    """
    k_mes = dft2(x_img)
    env = lowfreq_reconstruction_2D(k_mes, lf=0.08)
    mask = envelope_mask(env)
    mask = largest_component(mask)
    return mask
