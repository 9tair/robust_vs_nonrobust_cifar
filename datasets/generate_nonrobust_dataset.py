import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

class NonRobustDatasetGenerator:
    def __init__(self, epsilon=8/255, steps=10, alpha=2/255, checkpoint_path="results/datasets/nonrobust_checkpoint.pth"):
        """
        Generate a dataset with non-robust features using adversarial perturbations.
        :param epsilon: Maximum perturbation size
        :param steps: Number of attack steps
        :param alpha: Step size for iterative attack
        :param checkpoint_path: File path to store checkpoint
        """
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.start_index = 0  # Default start index

        # Load a standard model
        from models.get_model import ModelLoader
        self.model = ModelLoader(model_name="Standard").load_model().to(self.device).eval()

        # Check for an existing checkpoint
        if os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint if available."""
        checkpoint = torch.load(self.checkpoint_path)
        self.non_robust_images = checkpoint["non_robust_images"]
        self.labels = checkpoint["labels"]
        self.start_index = checkpoint["last_index"]
        print(f"✅ Resuming from checkpoint at index {self.start_index}")

    def save_checkpoint(self, non_robust_images, labels, last_index):
        """Ensure directory exists and save the current state as a checkpoint."""
        checkpoint_dir = "results/datasets"
        os.makedirs(checkpoint_dir, exist_ok=True)  # ✅ Ensure directory exists

        checkpoint_path = os.path.join(checkpoint_dir, "nonrobust_checkpoint.pth")
        torch.save({
            "non_robust_images": non_robust_images,
            "labels": labels,
            "last_index": last_index
        }, checkpoint_path)
        
        print(f"✅ Checkpoint saved at index {last_index}")

    def generate_adversarial_example(self, image, label):
        """
        Generate an adversarial example using PGD attack.
        """
        image = image.to(self.device)
        label = label.to(self.device)
        perturbed_image = image.clone().detach().requires_grad_(True)

        for _ in range(self.steps):
            output = self.model(perturbed_image)
            loss = F.cross_entropy(output, label)
            loss.backward()

            # PGD update step
            perturbation = self.alpha * perturbed_image.grad.sign()
            perturbed_image = perturbed_image + perturbation
            perturbed_image = torch.clamp(perturbed_image, min=image - self.epsilon, max=image + self.epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure valid pixel range
            perturbed_image = perturbed_image.detach().requires_grad_(True)

        return perturbed_image.detach()

    def process_dataset(self):
        """
        Process the dataset, generating adversarial examples and saving progress every 1000 images.
        If interrupted, it resumes from the last saved checkpoint.
        """
        # ✅ Ensure checkpoint and final save directory exists
        checkpoint_dir = "results/datasets"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Define paths
        checkpoint_path = os.path.join(checkpoint_dir, "nonrobust_checkpoint.pth")
        final_path = os.path.join(checkpoint_dir, "nonrobust_cifar10_final.pth")

        # Load CIFAR-10 dataset
        batch_size = 8
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        non_robust_images, labels = [], []
        start_index = 0  # Default starting index

        # ✅ Load checkpoint if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            non_robust_images = checkpoint["non_robust_images"]
            labels = checkpoint["labels"]
            start_index = checkpoint["last_index"]
            print(f"✅ Resuming from checkpoint at index {start_index}")

        # Process dataset from last checkpoint index
        for i, (image, label) in enumerate(dataloader):
            if i < start_index:  # Skip already processed images
                continue

            print(f"Processing image {i}/{len(dataloader.dataset)} on {self.device}")

            # Generate adversarial examples
            adversarial_image = self.generate_adversarial_example(image, label)
            non_robust_images.append(adversarial_image.cpu())
            labels.append(label)

            # ✅ Save checkpoint every 1000 images
            if i % 1000 == 0 and i > 0:
                torch.save({
                    "non_robust_images": non_robust_images,
                    "labels": labels,
                    "last_index": i
                }, checkpoint_path)
                print(f"✅ Checkpoint saved at index {i}")

        # ✅ Save final dataset
        torch.save((non_robust_images, labels), final_path)
        print(f"✅ Final dataset saved at {final_path}")

        # ✅ Remove checkpoint after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("✅ Checkpoint removed (completed successfully)")
        
if __name__ == "__main__":
    generator = NonRobustDatasetGenerator()
    generator.process_dataset()
