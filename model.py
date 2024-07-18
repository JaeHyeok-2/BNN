import gc
import cv2 as cv
import numpy as np 
import einops
from skimage import feature
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy, auroc
import timm
from fvcore.nn import FlopCountAnalysis, parameter_count
from BNext.src.bnext import BNext
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast 



class BNext4DFR(nn.Module):
    def __init__(self, num_classes, backbone='BNext-T', 
                 freeze_backbone=True, add_magnitude_channel=True, add_fft_channel=True, add_lbp_channel=True,
                 learning_rate=1e-4, pos_weight=1.):
        super(BNext4DFR, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # loads the backbone1
        self.backbone = backbone
        size_map = {"BNext-T": "tiny", "BNext-S": "small", "BNext-M": "middle", "BNext-L": "large"}
        if backbone in size_map:
            size = size_map[backbone]
            # loads the pretrained model
            self.base_model = nn.ModuleDict({"module": BNext(num_classes=1000, size=size)})
            # pretrained_state_dict = torch.load(f"pretrained/{size}_checkpoint.pth.tar", map_location="cpu")
            # self.base_model.load_state_dict(pretrained_state_dict)
            self.base_model = self.base_model.module
        else:
            print(backbone)
            raise ValueError("Unsupported Backbone!")
        
        # update the preprocessing metas
        assert isinstance(add_magnitude_channel, bool)
        self.add_magnitude_channel = add_magnitude_channel
        assert isinstance(add_fft_channel, bool)
        self.add_fft_channel = add_fft_channel
        assert isinstance(add_lbp_channel, bool)
        self.add_lbp_channel = add_lbp_channel
        self.new_channels = sum([self.add_magnitude_channel, self.add_fft_channel, self.add_lbp_channel])
        
        # loss parameters
        self.pos_weight = pos_weight
        
        if self.new_channels > 0:
            self.adapter = nn.Conv2d(in_channels=3+self.new_channels, out_channels=3, 
                                     kernel_size=3, stride=1, padding=1)
        else:
            self.adapter = nn.Identity()
            
        # disables the last layer of the backbone
        self.inplanes = self.base_model.fc.in_features
        self.base_model.deactive_last_layer=True
        self.base_model.fc = nn.Identity()
        # eventually freeze the backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False
        # add a new linear layer after the backbone
        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)
    
    def forward(self, x):
        if self.add_magnitude_channel or self.add_fft_channel or self.add_lbp_channel:
            x = self.add_new_channels(x)
        x_adapted = self.adapter(x)
        x_adapted = (x_adapted - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=x.device).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=x.device).view(1, -1, 1, 1)
        features = self.base_model(x_adapted)
        logits = self.fc(features)
        return logits
    
    def _add_new_channels_worker(self, image):
        gray = cv.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
        
        new_channels = []
        if self.add_magnitude_channel:
            new_channels.append(np.sqrt(cv.Sobel(gray,cv.CV_64F,1,0,ksize=7)**2 + cv.Sobel(gray,cv.CV_64F,0,1,ksize=7)**2) )
        
        if self.add_fft_channel:
            new_channels.append(20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-9))
        
        if self.add_lbp_channel:
            new_channels.append(feature.local_binary_pattern(gray, 3, 6, method='uniform'))
        new_channels = np.stack(new_channels, axis=2) / 255
        return torch.from_numpy(new_channels).to(image.device).float()
        
    def add_new_channels(self, images):
        images_copied = einops.rearrange(images, "b c h w -> b h w c")
        new_channels = torch.stack([self._add_new_channels_worker(image) for image in images_copied], dim=0)
        images_copied = torch.concatenate([images_copied, new_channels], dim=-1)
        images_copied = einops.rearrange(images_copied, "b h w c -> b c h w")
        return images_copied
    
class ModelCheckpoint:
    def __init__(self, monitor='val_acc', mode='max', save_top_k=1, filename='model_epoch_{epoch}_acc_{val_acc:.2f}_auc_{val_auc:.2f}.pt'):
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_model_path = None

    def __call__(self, epoch, model, metrics):
        score = metrics[self.monitor]
        if (self.mode == 'max' and score > self.best_score) or (self.mode == 'min' and score < self.best_score):
            self.best_score = score
            self.best_model_path = self.filename.format(epoch=epoch, **metrics)
            torch.save(model.state_dict(), self.best_model_path)
            print(f"Model saved to {self.best_model_path}")

def train_and_evaluate(model, train_loader, val_loader, num_epochs, accumulation_steps=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=model.learning_rate)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=5)
    scaler = GradScaler()

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', filename='model_epoch_{epoch}_train_acc_{train_acc:.2f}_val_acc_{val_acc:.2f}.pt')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        all_train_labels = []
        all_train_logits = []
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            images = batch["image"].to(device)
            labels = batch["is_real"][:, 0].float().to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = nn.functional.binary_cross_entropy_with_logits(outputs[:, 0], labels)
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item()
            train_logits = outputs[:, 0]
            all_train_labels.append(labels)
            all_train_logits.append(train_logits)
        
        scheduler.step()
        
        all_train_labels = torch.cat(all_train_labels)
        all_train_logits = torch.cat(all_train_logits)
        train_acc = accuracy(preds=all_train_logits, target=all_train_labels, task="binary", average="micro")
        train_auc = auroc(preds=all_train_logits, target=all_train_labels.long(), task="binary", average="micro")
        
        model.eval()
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_logits = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images = batch["image"].to(device)
                labels = batch["is_real"][:, 0].float().to(device)
                outputs = model(images)
                val_logits = outputs[:, 0]
                all_val_labels.append(labels)
                all_val_logits.append(val_logits)
        
        all_val_labels = torch.cat(all_val_labels)
        all_val_logits = torch.cat(all_val_logits)
        val_acc = accuracy(preds=all_val_logits, target=all_val_labels, task="binary", average="micro")
        val_auc = auroc(preds=all_val_logits, target=all_val_labels.long(), task="binary", average="micro")
        
        metrics = {
            'train_loss': running_loss / len(train_loader),
            'train_acc': train_acc.item(),
            'train_auc': train_auc.item(),
            'val_acc': val_acc.item(),
            'val_auc': val_auc.item()
        }
        
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_acc']:.4f}, Train AUC: {metrics['train_auc']:.4f}, Val Acc: {metrics['val_acc']:.4f}, Val AUC: {metrics['val_auc']:.4f}")

        
        checkpoint_callback(epoch, model, metrics)
    

if __name__ == "__main__":
    # Example usage
    # Assume `train_dataloader` and `test_dataloader` are predefined PyTorch DataLoader objects
    model = BNext4DFR(num_classes=2)
    x = torch.randn(8, 3, 224, 224)
    print(model, x.shape)
    # train_model(model, train_dataloader, num_epochs=10)
    # evaluate_model(model, test_dataloader)