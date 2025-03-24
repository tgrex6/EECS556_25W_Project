import torch
import kornia.metrics as metrics

def evaluate_image(pred_batch, target_batch, data_range=255.0):

    assert pred_batch.shape == target_batch.shape
    
    pred_batch = pred_batch.to(device=target_batch.device)
    
    ssim_val = metrics.ssim(
        pred_batch, 
        target_batch, 
        window_size=7,        
        max_val=data_range     
    )
    ssim_val = ssim_val.mean()
    
    mse = torch.mean((pred_batch - target_batch) ** 2, dim=[1, 2, 3])
    psnr_val = 10 * torch.log10(data_range**2 / mse).mean()
    
    return ssim_val.item(), psnr_val.item()