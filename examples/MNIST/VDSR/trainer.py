from torch.utils.data import DataLoader
import torch
from model import VDSR
import os
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
import argparse
import time
import ast

from data import PairedMNIST, transform_HRDS

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    i = 1
    
    for lr_up_batch,hr_batch in train_loader:
        batchsize = hr_batch.shape[0]
        

        hr_batch = hr_batch.to(device, non_blocking=True)
        lr_up_batch = lr_up_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        sr = model(lr_up_batch)
        loss = criterion(sr, hr_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * hr_batch.size(0)
        #print(f'{i*batchsize}/{len(train_loader.dataset)}    loss = {loss.item()}')
        i+=1

    train_loss /= len(train_loader.dataset)
    return train_loss


def validation_one_epoch(model, val_loader, criterion,device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    with torch.no_grad():
        for lr_up_batch, hr_batch, in val_loader:
            hr_batch = hr_batch.to(device, non_blocking=True)
            lr_up_batch = lr_up_batch.to(device, non_blocking=True)

            sr = model(lr_up_batch)

            loss = criterion(sr, hr_batch)
            val_loss += loss.item() * hr_batch.size(0)

            val_psnr += psnr(sr, hr_batch) * hr_batch.size(0)

    val_loss /= len(val_loader.dataset)
    val_psnr /= len(val_loader.dataset)

    # Step scheduler using validation loss
    scheduler.step(val_loss)
    return val_loss, val_psnr

    

def psnr(pred, target, max_val=3.0):
    """
    Compute PSNR for grayscale images.
    pred, target: (B, C, H, W), values in [0,1]
    """
    mse = torch.mean((pred - target) ** 2, dim=[1,2,3])  # per image
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4,help='learning rate')
    parser.add_argument('--loss', type=str, default='l1',help='loss type')
    parser.add_argument('--n_epochs', type=int, default=300,help='number of epochs')
    parser.add_argument('--train_prop', type=float, default=0.7,help='Proportion training/val')
    parser.add_argument('--save_dir', type=str, default='examples/MNIST/VDSR/ckp',help='save directory')
    parser.add_argument('--batch_size', type=int, default=16,help='Batch size')
    parser.add_argument('--n_filters', type=int, default=64,help='number of filters in the model')
    parser.add_argument('--n_resblocks', type=int, default=20,help='number of resblocks')
    parser.add_argument('--remove_digits',type = ast.literal_eval,default=[],help='Remove some numbers from the dataset')
    parser.add_argument('--noise_level', type=float, default=0.0,help='Noise level')
    parser.add_argument('--noise_type', type=str, default='None',help='noise type')
    args = parser.parse_args()

    lr = args.lr
    loss = args.loss
    n_epochs = args.n_epochs
    train_proportion = args.train_prop
    save_dir = args.save_dir
    batch_size = args.batch_size

    num_filters= args.n_filters
    num_resblocks= args.n_resblocks

    digits_to_remove = args.remove_digits
    noise_type = args.noise_type
    noise_level = args.noise_level

    if noise_type == 'None':
        noise_type = None
    assert noise_type in ['poisson', 'gauss', None], f'Noise type should be among "poisson", "gauss", None'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses = []
    val_losses = []
    val_psnrs = []
    best_psnr = 0.0
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)

    # DEFINE DATASET AND DLOADER
    

    MNIST_folder = '/localhome/iaga_dv/Dokumente/MNIST_data'
    MNIST_pair_ds = PairedMNIST(root = MNIST_folder,transform_Y= transform_HRDS(noise_type=noise_type, noise_level=noise_level),download=True)
    n_img = len(MNIST_pair_ds)

    # For each digit get the indexes associated with that label
    index_digit = [[], [], [], [], [], [], [], [], [], []]
    for i in range(n_img):
        label = MNIST_pair_ds.targets[i]
        index_digit[label].append(i)
    
    idx_to_remove = []
    for digit in digits_to_remove:
        idx_to_remove += index_digit[digit]

        


    MNIST_pair_ds_train = Subset(MNIST_pair_ds, list(set(range(int(train_proportion* n_img)))-set(idx_to_remove)) )
    MNIST_pair_ds_val = Subset(MNIST_pair_ds, list(set(range(int(train_proportion* n_img), n_img))- set(idx_to_remove)))

    loader_train = DataLoader(MNIST_pair_ds_train, batch_size = batch_size, shuffle =True, num_workers=4, pin_memory=True)
    loader_val = DataLoader(MNIST_pair_ds_val, batch_size = batch_size, shuffle =False, num_workers=4, pin_memory=True)

    model = VDSR(num_filters=num_filters, num_resblocks=num_resblocks).to(device)

    if loss == 'l2':
        criterion = torch.nn.MSELoss()
    elif loss == 'l1':
        criterion = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    t0 = time.time()
    for epoch in range(n_epochs):
        t1 = time.time()
        train_loss = train_one_epoch(model, loader_train, optimizer, criterion, device)
        val_loss, val_psnr = validation_one_epoch(model, loader_val, criterion, device)

        #best_psnr = torch.tensor(val_psnrs).max()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr.item())


        torch.save(model.state_dict(), os.path.join(save_dir, 'models',"model_latest.pth"))

        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(),os.path.join(save_dir, 'models', 'model_best.pth'))

        t2 = time.time()
        print(f"Epoch [{epoch}/{n_epochs}] "f"Train Loss: {train_loss:.6f} "
          f"Val Loss: {val_loss:.6f} "
          f"Val PSNR: {val_psnr:.2f} dB "
          f"LR: {optimizer.param_groups[0]['lr']:.6e}"
          f"Took {t2-t0:.2e}s for the training and {t2-t1:.2e}s for this epoch")
        

        # SAVING TRAINING LOSS CURVE
        plt.figure()
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(f'Training Loss ({loss})')
        plt.title('Loss per Epoch')
        plt.grid(True)

        save_path = os.path.join(save_dir, 'plots',f'loss_train.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # SAVING VAL LOSS CURVE
        plt.figure()
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(f'Validation Loss ({loss})')
        plt.title('Loss per Epoch')
        plt.grid(True)

        save_path = os.path.join(save_dir, 'plots',f'loss_val.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # SAVING VAL PSNR CURVE
        plt.figure()
        plt.plot(range(1, len(val_psnrs) + 1), val_psnrs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(f'Validation PSNR ')
        plt.title('PSNR per Epoch')
        plt.grid(True)

        save_path = os.path.join(save_dir, 'plots',f'PSNR_val.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
