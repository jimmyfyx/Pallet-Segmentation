import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dataset import WarehouseSegDataset
from dinovit import DINOv2DPT

# torch.autograd.set_detect_anomaly(True)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.0):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.0 * intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.0):
        inputs = inputs.reshape(inputs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return 1 - dice

class Trainer:
    def __init__(self, run_idx, lr, batch_size, num_epochs) -> None:
        self.run_idx = run_idx

        # Hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

        # Dataset
        self.train_dataset = WarehouseSegDataset(data_type='train')
        self.val_dataset = WarehouseSegDataset(data_type='val')
        self.test_dataset = WarehouseSegDataset(data_type='test')

        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Model, loss function, optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINOv2DPT(output_size=(416, 416)).to(self.device)
        self.dbce_loss = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Logger
        self.logger_path = f'./runs/logs/{self.run_idx}'
        if not os.path.exists(self.logger_path):
            os.makedirs(self.logger_path, exist_ok=True)
        self.logger = SummaryWriter(self.logger_path)
    
    def save_model(self, epoch, train_loss, validation_loss):
        decoder_save_dir = f'./checkpoints/{self.run_idx}/decoder/'
        if not os.path.exists(decoder_save_dir):
            os.makedirs(decoder_save_dir, exist_ok=True)
        decoder_save_path = os.path.join(decoder_save_dir, f'epoch{epoch + 1}_{train_loss:.6f}_{validation_loss:.6f}.pth')
        torch.save(self.model.decoder.state_dict(), decoder_save_path)

        final_layer_save_dir = f'./checkpoints/{self.run_idx}/final_layer'
        if not os.path.exists(final_layer_save_dir):
            os.makedirs(final_layer_save_dir, exist_ok=True)
        final_layer_save_path = os.path.join(final_layer_save_dir, f'epoch{epoch + 1}_{train_loss:.6f}_{validation_loss:.6f}.pth')
        torch.save(self.model.final_layer.state_dict(), final_layer_save_path)

    def plot_training_curve(self, train_losses, validation_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(self.logger_path + '/loss_curve.png')

    def train(self):
        # Training loop
        print('Start Training ...')
        train_losses = []
        validation_losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_loader = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Train]')
            for rgb_image, mask in train_loader:
                rgb_image = rgb_image.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(rgb_image)
                dbce_loss = self.dbce_loss(output, mask)
                dbce_loss.backward()
                self.optimizer.step()

                train_loss += dbce_loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.logger.add_scalar('Loss/train', avg_train_loss, epoch)

            # Validation
            print('Start Validation ...')
            self.model.eval()
            with torch.no_grad():
                validation_loss = 0.0
                validation_loader = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Validation]')
                for rgb_image, mask in validation_loader:
                    rgb_image = rgb_image.to(self.device)
                    mask = mask.to(self.device)

                    output = self.model(rgb_image)
                    dbce_loss = self.dbce_loss(output, mask)
                    
                    validation_loss += dbce_loss.item()

                avg_validation_loss = validation_loss / len(self.val_loader)
                validation_losses.append(avg_validation_loss)
                self.logger.add_scalar('Loss/validation', avg_validation_loss, epoch)

            print(f'Epoch [{epoch + 1}/{self.num_epochs}]| Training Loss: {avg_train_loss:.6f} | Validation Loss: {avg_validation_loss:.6f}')

            if (epoch + 1) % 5 == 0:
                self.save_model(epoch, avg_train_loss, avg_validation_loss)  # Save model
            
        self.logger.close()
        self.plot_training_curve(train_losses, validation_losses)
    
    def test(self, decoder_path, final_layer_path):
        self.model.decoder.load_state_dict(torch.load(decoder_path))
        self.model.final_layer.load_state_dict(torch.load(final_layer_path))
        self.model.eval()
        test_loss = 0.0
        test_loader = tqdm(self.test_loader, desc='Test')
        with torch.no_grad():
            for rgb_image, mask in test_loader:
                rgb_image = rgb_image.to(self.device)
                mask = mask.to(self.device)

                output = self.model(rgb_image)
                dbce_loss = self.dbce_loss(output, mask)
                
                test_loss += dbce_loss.item()

        avg_test_loss = test_loss / len(self.test_loader)
        print(f'Test Loss: {avg_test_loss:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, help='training num')

    # Training
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=40, type=int, help='epochs to train')
    
    args = parser.parse_args()

    trainer = Trainer(args.runs, args.lr, args.batch_size, args.num_epochs)
    trainer.train()



