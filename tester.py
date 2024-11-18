import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
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

class Tester:
    def __init__(self, run_idx, batch_size) -> None:
        self.run_idx = run_idx

        # Hyperparameters
        self.batch_size = batch_size

        # Dataset
        self.test_dataset = WarehouseSegDataset(data_type='test')

        # Dataloader
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Model and loss function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DINOv2DPT(output_size=(416, 416)).to(self.device)
        self.dbce_loss = DiceBCELoss()

        self.save_dir = f'./test_results/{run_idx}'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
    
    @staticmethod
    def denormalize_image(image):
        mean = (123.675, 116.28, 103.53)
        std = (58.395, 57.12, 57.375)

        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        image = image * std + mean
        return image

    @staticmethod
    def calculate_batch_iou(ground_truth, predicted, num_classes=2):
        ground_truth = ground_truth.cpu().numpy().astype(np.uint8)
        predicted = predicted.cpu().numpy()
        predicted = (predicted > 0.5).astype(np.uint8)  # Convert predict mask to binary

        batch_size = ground_truth.shape[0]
        iou_per_class = {class_idx: [] for class_idx in range(num_classes)}

        for batch_idx in range(batch_size):
            for class_idx in range(num_classes):
                # Extract the binary masks for the current batch and class
                gt = ground_truth[batch_idx, class_idx]  # Ground truth for this class
                pred = predicted[batch_idx, class_idx]  # Prediction for this class

                # Calculate Intersection and Union
                intersection = np.logical_and(gt, pred).sum()
                union = np.logical_or(gt, pred).sum()

                # Handle edge case where union is 0
                if union == 0:
                    iou = 1.0 if intersection == 0 else 0.0
                else:
                    iou = intersection / union

                # Append IoU for the current class
                iou_per_class[class_idx].append(iou)

        mean_iou_per_class = {class_idx: np.mean(iou_list) for class_idx, iou_list in iou_per_class.items()}
        return mean_iou_per_class

    def save_result(self, rgb_image, predicted_mask, ground_truth_mask, count):
        # print(rgb_image.shape)
        # print(mask.shape)
        
        # Denormalize the RGB image and scale back to [0, 1]
        rgb_image = self.denormalize_image(rgb_image.cpu()).clamp(0, 255) / 255.0
        rgb_image = rgb_image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) format

        # Convert masks to NumPy arrays
        predicted_mask = predicted_mask.cpu().numpy()
        ground_truth_mask = ground_truth_mask.cpu().numpy()

        # Resize the RGB image to match the mask resolution
        rgb_image_resized = cv2.resize(rgb_image, (predicted_mask.shape[2], predicted_mask.shape[1]), interpolation=cv2.INTER_LINEAR)

        # Function to create a colored mask overlay
        def create_colored_mask(mask):
            colored_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.float32)
            colored_mask[mask[0] > 0.5, 0] = 1.0  # Red for ground
            colored_mask[mask[1] > 0.5, 1] = 1.0  # Green for pallet
            return colored_mask

        # Create overlays for predicted and ground truth masks
        predicted_colored_mask = create_colored_mask(predicted_mask)
        ground_truth_colored_mask = create_colored_mask(ground_truth_mask)

        predicted_overlay = cv2.addWeighted(rgb_image_resized, 0.7, predicted_colored_mask, 0.3, 0)
        ground_truth_overlay = cv2.addWeighted(rgb_image_resized, 0.7, ground_truth_colored_mask, 0.3, 0)

        # Create a figure with two subplots
        plt.figure(figsize=(12, 6))

        # Plot the predicted overlay
        plt.subplot(1, 2, 1)
        plt.imshow(predicted_overlay)
        plt.title("Predicted Mask Overlay")
        plt.axis("off")

        # Plot the ground truth overlay
        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth_overlay)
        plt.title("Ground Truth Mask Overlay")
        plt.axis("off")

        # Save the combined figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{count:05}.png'))
        plt.clf()

    def test(self, decoder_ckpt, final_layer_ckpt):
        decoder_path = f'./checkpoints/{self.run_idx}/decoder/{decoder_ckpt}'
        final_layer_path = f'./checkpoints/{self.run_idx}/final_layer/{final_layer_ckpt}'
        self.model.decoder.load_state_dict(torch.load(decoder_path))
        self.model.final_layer.load_state_dict(torch.load(final_layer_path))
        self.model.eval()

        test_loss = 0.0
        num_classes = 2
        iou_per_class = {class_idx: [] for class_idx in range(num_classes)}
        test_loader = tqdm(self.test_loader, desc='Test')
        with torch.no_grad():
            for i, (rgb_image, mask) in enumerate(test_loader):
                rgb_image = rgb_image.to(self.device)
                mask = mask.to(self.device)

                output = self.model(rgb_image)
                dbce_loss = self.dbce_loss(output, mask)
                test_loss += dbce_loss.item()

                mean_iou_per_class = self.calculate_batch_iou(mask, output, num_classes=2)   
                for class_idx, iou in mean_iou_per_class.items():
                    iou_per_class[class_idx] .append(iou)             
                self.save_result(rgb_image[0], output[0], mask[0], i)  # Save sample result

        avg_test_loss = test_loss / len(self.test_loader)
        print(f'Test Loss: {avg_test_loss:.6f}')
        for class_idx, iou_list in iou_per_class.items():
            print(f"Mean IoU for class {class_idx}: {np.mean(iou_list):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, help='training num')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--decoder_ckpt', default=None, type=str, help='saved decoder weights .pth file')
    parser.add_argument('--final_layer_ckpt', default=None, type=str, help='saved final_layer .pth file')
    args = parser.parse_args()

    trainer = Tester(args.runs, args.batch_size)
    trainer.test(args.decoder_ckpt, args.final_layer_ckpt)



