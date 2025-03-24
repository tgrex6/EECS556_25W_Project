import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import glob


class DenoisingDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 transform=None, 
                 mode='train', 
                 patch_size=224, 
                 stride=224):
        self.mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        
        self.samples = []

        if self.mode == 'train':
            subfolders = glob.glob(os.path.join(data_root, '*'))
            for subdir in subfolders:
                if os.path.isdir(subdir):
                    gt_files = glob.glob(os.path.join(subdir, 'GT*.*'))
                    noisy_files = glob.glob(os.path.join(subdir, 'NOISY*.*'))
                    
                    for gt_path in gt_files:
                        base_name = os.path.basename(gt_path).split('GT')[-1]
                        noisy_path = os.path.join(subdir, 'NOISY' + base_name)
                        
                        if noisy_path in noisy_files:
                            with Image.open(gt_path) as gt_img:
                                w, h = gt_img.size

                            for top in range(0, h - patch_size + 1, stride):
                                for left in range(0, w - patch_size + 1, stride):
                                    x1, y1 = left, top
                                    x2, y2 = left + patch_size, top + patch_size
                                    self.samples.append((gt_path, noisy_path, x1, y1, x2, y2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt_path, noisy_path, x1, y1, x2, y2 = self.samples[idx]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_patch = gt_img.crop((x1, y1, x2, y2))
        
        noisy_img = Image.open(noisy_path).convert('RGB')
        noisy_patch = noisy_img.crop((x1, y1, x2, y2))

        seed = torch.seed()
        torch.manual_seed(seed)
        if self.transform is not None:
            gt_patch = self.transform(gt_patch)
        torch.manual_seed(seed)
        if self.transform is not None:
            noisy_patch = self.transform(noisy_patch)

        return noisy_patch, gt_patch


if __name__ == "__main__":
    data_path = 'D:\wavevit\SIDD_Small_sRGB_Only\Data'
    dataset = DenoisingDataset(data_path, mode='train')
    print(len(dataset))
