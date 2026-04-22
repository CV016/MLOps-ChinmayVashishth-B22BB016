import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import UNet
from dataset import get_dataloaders
from metrics import calculate_metrics

def plot_and_save(history, save_path="training_metrics.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='red')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['val_miou'], label='mIoU', color='blue')
    ax2.plot(epochs, history['val_mdice'], label='mDice', color='green')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_pipeline(data_dir, epochs=15, batch_size=8, lr=1e-3, device="cpu"):
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    model = UNet(in_channels=3, num_classes=23).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_miou': [], 'val_mdice': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        model.eval()
        epoch_miou, epoch_mdice = 0.0, 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                
                miou, mdice = calculate_metrics(outputs, masks, num_classes=23)
                epoch_miou += miou
                epoch_mdice += mdice

        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_miou'].append(epoch_miou / len(val_loader))
        history['val_mdice'].append(epoch_mdice / len(val_loader))

    plot_and_save(history)
    torch.save(model.state_dict(), "unet_cityscapes.pth")
    return history