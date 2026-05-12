# On Hallucinations in Inverse Problems: Fundamental Limits and Computable Bounds

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

---

This repository provides the algorithms for reproducing the experiments described in the paper **On Hallucinations in Inverse Problems: Fundamental Limits and Computable Bounds** including applications in **Super-resolution of Sentinel 2 data, MRI acceleration, and MNIST super-resolution**. The code is based on the library for [Kernel Sizes computations](https://github.com/nm19000/AccuracyBounds).
´
Python version : Python 3.10.12


Setting up the environments with venv : 
```bash
git clone https://github.com/davidiagraid/AccuracyBounds_private.git
python -m venv env_MRI_MNIST
python -m venv env_S2SR

source env_MRI_MNIST/bin/activate
cd AccuracyBounds/examples/MNIST
pip install -r requirements.txt
cd AccuracyBounds/examples/MRI_code
pip install -r requirements.txt

source env_S2SR/bin/activate
cd AccuracyBounds/examples/S2_SR
pip install -r requirements.txt

```

Experiments on super-resolution of sentinel 2 data require a separate environment from MRI acceleration and MNIST super-resolution.

## Data Preparation

### For all experiments:
- Download the necessary datasets:
  - **Satellite Super-Resolution (S2 SR)**: Download from [Hugging Face Dataset](https://huggingface.co/datasets/isp-uv-es/opensr-test).
  - **MRI Acceleration**: Download from the official website of [Fast MRI](https://fastmri.med.nyu.edu/).
  - **MNIST Super-Resolution**: Automatically downloaded by `torchvision`.

## Applications

### 1. Satellite Super-Resolution (S2 SR)

#### Data Structure

```
cross_processed/
├── naip/
│   ├── 1/
│   │   ├── hr_res.tif
│   │   ├── lr_res.tif
│   │   ├── sr_res.tif
│   ├── 2/
│   ├── ...
├── spain_crops/
├── spain_urban/
├── spot/
```

#### Data Preparation

- Download data from [Hugging Face Dataset](https://huggingface.co/datasets/isp-uv-es/opensr-test).
- Set the data under the above structure
- Use `utils.py` to generate patched datasets if needed.

#### Running Experiments

- **Kernel Size Computations:**
  (not necessary)
  ```bash
  python examples/S2_SR/Kernelsize_computations.py
  ```
- **Operator Calculation:**
  ```bash
  python examples/S2_SR/op_testing.py
  ```
- **Detail pasting method:**
  ```bash
  python examples/S2_SR/det_pasting.py
  ```
- **Notebook for other experiments:**
  Launch the notebook
  ```bash
  examples/S2SR/playground.ipynb
  ```

### 2. MRI Acceleration


#### Preliminary steps
- Download the brain multicoil data from the official website of [Fast MRI](https://fastmri.med.nyu.edu/)
- Process the data and save it under single coil format, running the part "Get dataset information and save Single Coil Brain MRI data" of the notebook at "examples/MRI_code/playground.py" (expected time : 1 or 2 hours on for CPUs for the whole dataset)
- Download the pre-trained U-Net Model weights following the instructions given in the [Fast MRI library](https://github.com/facebookresearch/fastMRI/tree/main/fastmri_examples/unet). Adapt the path to the model's weights in the `load_model` function of examples/MRI_code/utils.py .


#### Running Experiments

- **Any detail pasting:**
  ```bash
  python examples/MRI_code/det_pasting.py
  ```

  - **Anomaly pasting method:**
  Before running the python script, put together the slices containing anomalies ,in the same folder. Each slice needs to be stored in a .npy file, they should be named 'idx_{index}.npy'. The index numbers need to correspond to the order given in the file info_files/annotation_info_v0.json (in the list associated to the dictionnaryunder the key 'slicelist'. Each slice is determined by the scan name and its slice number in that scan.)

  ```bash
  python examples/MRI_code/anomaly_pasting.py
  ```
  
- **Notebook for other experiments:** `examples/MRI_code/playground.ipynb`

---

### 3. MNIST Super-Resolution

- **Notebook for experiments:** `examples/MNIST/playground.ipynb`
- **Training VDSR Models:**
  ```bash
  python examples/MNIST/VDSR/trainer.py
  ```

---

## How to Run Experiments

### 1. Super-Resolution of Sentinel 2 Data (examples/S2SR)

#### Requirements
- Data: [Download here](https://huggingface.co/datasets/isp-uv-es/opensr-test)

#### Preliminary steps
- Compute the Null space operator under matrix form, running op_testing.py . Ensure you adapt the paths to store them : DATA_FOLDER, and OPERATORS_FOLDER

#### Detail pasting
- Run det_pasting.py . Ensure you replace the default paths for storing the dataset as well as the pasted details (dataset_folder, folder_drawing_pairs)

#### Main experiments
- Run the notebook playground.py, and ensure you set the path variables corresponding to where you store your data



#### Parameters
   Parameter | Description | Default |
 |-----------|-------------|---------|
 | `PS_X` | [Patch size for High resolution images. It needs to be a multiple of 4] | `16` |
 | `p_X`, `p_Y` | [Order of the norm for the distance computations] | `1,2` |
 | `batch_size` | [For the Decoder-agnostic method computations] | `10` |
 | `noise_level` | [Normalized noise level for Decoder-agnostic method ] | `4000` |
 | `n_preds` | [Number of predictions for the theorem verification] | `10` |
 | `light_load` | [Possibility to load the patch dataset with minimum image openings for Kernelsize computations] | `True` |
---

### 2. MRI Acceleration (examples/MRI_code/)

#### Requirements
- Data: [Fast MRI](https://fastmri.med.nyu.edu/)
- Anomaly annotations and segmentation in the folder `experiments/MRI_code/info_files`

#### Preliminary steps
- **Prepare data** as mentionned above (expected time : 1 or 2 hours on for CPUs for the whole dataset)

#### Anomaly pasting
- Run anomaly_pasting.py . Ensure you change the paths of the dataset and the anomaly segnentation information according to your configuration.

#### Detail pasting
- Run det_pasting.py.py . Ensure you change the paths of the dataset according to your configuration.

#### Main experiments
- Run the notebook playground.py, and ensure you set the path variables corresponding to where you store your data



#### Parameters
 | Parameter | Description | Default |
 |-----------|-------------|---------|
 | `p_X`, `p_Y` | [Order of the norm for the distance computations] | `1,2` |
 | `SNR` | [Determines the noise level kept when applying the null-space projection] | `50` |
 | `acceleration_rate` | [Fraction of bands to keep in the forward model vertical subsampling] | `8` |
 | `Number of central bands` | [Number of central bands in the vertical subsampling (less than width//acceleration)] | `22` |
---

### 3. MNIST Super-Resolution (experiments/MNIST)

#### Requirements
- Data: Automatically downloaded by torchvision

#### Preliminary steps
- Train VDSR Models with different noise levels, running VDSR/trainer.py . We also put online model's weights if you prefer to skip the training.
Adjustable parameters :

| Parameter         | Description                          | Default                   |
| ----------------- | ------------------------------------ | ------------------------- |
| `--lr`            | learning rate                        | `2e-4`                    |
| `--loss`          | loss type                            | `l1`                      |
| `--n_epochs`      | number of epochs                     | `300`                     |
| `--train_prop`    | Proportion training/val              | `0.7`                     |
| `--save_dir`      | save directory                       | `examples/MNIST/VDSR/ckp` |
| `--batch_size`    | Batch size                           | `16`                      |
| `--n_filters`     | number of filters in the model       | `64`                      |
| `--n_resblocks`   | number of resblocks                  | `20`                      |
| `--remove_digits` | Remove some numbers from the dataset | `[]`                      |
| `--noise_level`   | Noise level                          | `0.0`                     |
| `--noise_type`    | noise type                           | `poisson`                 |
---
Expected training time : 3 hours locally on a standard lenovo thinkpad computer
#### Main experiments
- Run the notebook playground.py, and ensure you set the path variables corresponding to where you store your data

## Troubleshooting

- **Error**: `ModuleNotFoundError: No module named 'X'`
  - **Solution**: Ensure all dependencies are installed by running `pip install -r requirements.txt` in the activated virtual environment.

- **Error**: `CUDA out of memory`
  - **Solution**: Reduce `batch_size` or ensure no other processes are using the GPU.

- **Error**: `Data not found`
  - **Solution**: Verify that data is correctly downloaded and placed according to the structure outlined above.

## Citing Our Work

If you use this library in your research, please cite:

```bibtex
@article{XXX
}
```

---

## License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## Contact

For questions or feedback, please open an issue or contact [david.iagaru@polytechnique.edu].