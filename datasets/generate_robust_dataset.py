import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.get_model import ModelLoader
from models.feature_extractor import FeatureExtractor
import os

# **Force PyTorch to use GPU 3 (third GPU)**
torch.cuda.set_device(3)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# **Print GPU details before running**
print(f"Using GPU: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

# Function to print current memory usage
def print_memory_usage(msg=""):
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved(device) / 1024**3  # Convert to GB
    print(f"[{msg}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class RobustDatasetGenerator:
    def __init__(self, steps=1000, lr=0.1, checkpoint_path="results/datasets/robust_checkpoint.pth"):
        self.steps = steps
        self.lr = lr
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.start_index = 0  # Default starting index

        self.model = ModelLoader().load_model().to(self.device).eval()
        self.feature_extractor = FeatureExtractor(self.model).to(self.device).eval()

        # Check for an existing checkpoint
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint if available."""
        checkpoint = torch.load(self.checkpoint_path)
        self.robust_images = checkpoint["robust_images"]
        self.labels = checkpoint["labels"]
        self.start_index = checkpoint["last_index"]
        print(f"✅ Resuming from checkpoint at index {self.start_index}")

    def save_checkpoint(self, robust_images, labels, last_index):
        """Ensure directory exists and save the current state as a checkpoint."""
        checkpoint_dir = "results/datasets"
        os.makedirs(checkpoint_dir, exist_ok=True)  # ✅ Ensure directory exists

        checkpoint_path = os.path.join(checkpoint_dir, "robust_checkpoint.pth")
        torch.save({
            "robust_images": robust_images,
            "labels": labels,
            "last_index": last_index
        }, checkpoint_path)
    
        print(f"✅ Checkpoint saved at index {last_index}")

    def generate_robust_image(self, original_image, init_image):
        """Optimize init_image to match the robust features of original_image."""
        original_image, init_image = original_image.to(self.device), init_image.to(self.device)

        # Print memory usage before feature extraction
        print_memory_usage("Before feature extraction")
        target_features = self.feature_extractor(original_image)
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
        """
        Process the dataset, generating robust images and saving progress every 1000 images.
        If interrupted, it resumes from the last saved checkpoint.
        """
        # ✅ Ensure checkpoint and final save directory exists
        checkpoint_dir = "results/datasets"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define paths
        checkpoint_path = os.path.join(checkpoint_dir, "robust_checkpoint.pth")
        final_path = os.path.join(checkpoint_dir, "robust_cifar10_final.pth")

        # Load CIFAR-10 dataset
        batch_size = 8  # Adjust based on memory
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        robust_images, labels = [], []
        start_index = 0  # Default starting index

        # ✅ Load checkpoint if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            robust_images = checkpoint["robust_images"]
            labels = checkpoint["labels"]
            start_index = checkpoint["last_index"]
            print(f"✅ Resuming from checkpoint at index {start_index}")

        # Process dataset from last checkpoint index
        for i, (image, label) in enumerate(dataloader):
            if i < start_index:  # Skip already processed images
                continue

            print(f"Processing image {i}/{len(dataloader.dataset)} on {self.device}")

            # Generate robust image
            init_image, _ = next(iter(dataloader))  # Use a random image as init
            robust_image = self.generate_robust_image(image, init_image)
            robust_images.append(robust_image.cpu())
            labels.append(label)

            # ✅ Save checkpoint every 1000 images
            if i % 1000 == 0 and i > 0:
                torch.save({
                    "robust_images": robust_images,
                    "labels": labels,
                    "last_index": i
                }, checkpoint_path)
                print(f"✅ Checkpoint saved at index {i}")

        # ✅ Save final dataset
        torch.save((robust_images, labels), final_path)
        print(f"✅ Final dataset saved at {final_path}")

        # ✅ Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("✅ Checkpoint removed (completed successfully)")

if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear unused GPU memory
    print_memory_usage("Before running script")

    generator = RobustDatasetGenerator()
    generator.process_dataset()

    print_memory_usage("After running script")
