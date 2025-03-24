import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from wavevit import WaveViT
from model import DenoiseDecoder
from dataset import DenoisingDataset
from compound_loss import CompoundLoss
from torchvision import transforms
from utils import evaluate_image
from tqdm.auto import tqdm, trange
import wandb
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def initialize_model(device):
    encoder = WaveViT()
    decoder = DenoiseDecoder()
    model = nn.Sequential(encoder, decoder).to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    loss_fun = CompoundLoss().to(device) 
    
    return model, optimizer, scheduler, loss_fun

def evaluate_model(model, dataloader, loss_fun, device='cuda'):
    model.eval()
    model.to(device)
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for noisy_imgs, gt_imgs in tqdm(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            gt_imgs = gt_imgs.to(device)
            
            outputs = model(noisy_imgs)
            
            batch_loss = loss_fun(outputs, gt_imgs)
            total_loss += batch_loss.item() * noisy_imgs.size(0)  # 按样本数加权
            
            all_preds.append(outputs)
            all_targets.append(gt_imgs)

    preds = torch.cat(all_preds, dim=0)  # [N, C, H, W]
    targets = torch.cat(all_targets, dim=0)

    avg_ssim, avg_psnr = evaluate_image(preds, targets, data_range=255.0)
    avg_loss = total_loss / len(dataloader.dataset)

    return avg_ssim, avg_psnr, avg_loss

def train_model(data_root, num_epochs=1, batch_size=16, device='cuda'):
    set_seed(42)
    
    dataset = DenoisingDataset(
        data_root=data_root,
        mode='train',
        patch_size=224,
        stride=224,
        transform=transforms.ToTensor()
    )
    
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Training on {train_size} samples, validating on {val_size} samples.")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model, optimizer, scheduler, loss_fun = initialize_model(device)
    
    run = wandb.init(mode="offline",  # Change to "online" to enable logging
                    project="eecs556_project",
                    name = "denoising",
                        config = {
                            "num_epochs": num_epochs,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "batch_size": batch_size
                        })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    best_ssim = 0.0
    best_psnr = 0.0

    for epoch in trange(num_epochs, desc="Epoch"):
        model.train()
        train_loss = 0.0
        
        for step, (noisy_imgs, gt_imgs) in enumerate(tqdm(train_loader)):
            noisy_imgs = noisy_imgs.to(device)
            gt_imgs = gt_imgs.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(noisy_imgs)
            loss = loss_fun(outputs, gt_imgs)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * noisy_imgs.size(0)
        
            if (step + 1) % 5 == 0:
                wandb.log({"train_loss": train_loss / (step + 1)}, step=step+epoch*len(train_loader))
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {train_loss / (step + 1):.4f}")
        
            if (step + 1) % 10 == 0:            
                ssim, psnr, val_loss = evaluate_model(model, val_loader, loss_fun, device)
                wandb.log({"val_ssim": ssim, "val_psnr": psnr, "val_loss": val_loss}, step=step+epoch*len(train_loader))
                print(f"Validation - SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, Loss: {val_loss:.4f}")
                if ssim > best_ssim:
                    best_ssim = ssim
                    best_psnr = psnr
                    # save the best model with timestamp to avoid overwriting
                    torch.save(model.state_dict(), f"best_model_{timestamp}.pth")
                model.train()



        scheduler.step()
    
    run.finish()
    print(f"Best SSIM: {best_ssim:.4f}, Best PSNR: {best_psnr:.4f}")

if __name__ == "__main__":
    data_root = 'D:\wavevit\SIDD_Small_sRGB_Only\Data'
    wandb.login()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(data_root, num_epochs=1, batch_size=4, device=device)
    
