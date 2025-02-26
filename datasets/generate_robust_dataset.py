import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from models.get_model import ModelLoader
from models.feature_extractor import FeatureExtractor
import os

# 1️**Force PyTorch to use GPU 2 (third GPU)**
torch.cuda.set_device(3)
device = torch.device("cuda:3")

# 2️**Print GPU details before running**
print(f"Using GPU: {torch.cuda.get_device_name(device)}")
print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

# Function to print current memory usage
def print_memory_usage(msg=""):
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
    print(f"[{msg}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class RobustDatasetGenerator:
    def __init__(self, steps=1000, lr=0.1):
        self.steps = steps
        self.lr = lr
        self.model = ModelLoader().load_model().to(device).eval()
        self.feature_extractor = FeatureExtractor(self.model).to(device).eval()

    def generate_robust_image(self, original_image, init_image):
        """Optimize init_image to match the robust features of original_image."""
        original_image, init_image = original_image.to(device), init_image.to(device)

        # Print memory usage before feature extraction
        print_memory_usage("Before feature extraction")

        target_features = self.feature_extractor(original_image)

        # Print memory usage after feature extraction
        print_memory_usage("After feature extraction")

        init_image.requires_grad = True
        optimizer = torch.optim.Adam([init_image], lr=self.lr)

        for step in range(self.steps):
            optimizer.zero_grad()
            current_features = self.feature_extractor(init_image)
            loss = F.mse_loss(current_features, target_features)
            loss.backward(retain_graph=True)
            optimizer.step()
            init_image.data.clamp_(0, 1)

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        return init_image.detach()

    def process_dataset(self):
        # Reduce batch size to avoid memory issues
        batch_size = 8  # Lowered from default to reduce memory usage

        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        robust_images, labels = [], []

        for i, (image, label) in enumerate(dataloader):
            print_memory_usage(f"Processing Image {i}")

            init_image, _ = next(iter(dataloader))  # Use a random image as init
            robust_image = self.generate_robust_image(image, init_image)
            robust_images.append(robust_image.cpu())
            labels.append(label)

            print_memory_usage(f"After processing Image {i}")

        torch.save((robust_images, labels), "results/datasets/robust_cifar10.pth")
        print("Robust dataset saved!")

if __name__ == "__main__":
    # Clear unused GPU memory before starting
    torch.cuda.empty_cache()
    print_memory_usage("Before running script")

    generator = RobustDatasetGenerator()
    generator.process_dataset()

    print_memory_usage("After running script")
