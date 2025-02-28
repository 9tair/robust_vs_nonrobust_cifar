import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from models.get_model import ModelLoader
import os

class NonRobustDatasetGenerator:
    def __init__(self, epsilon=8/255, steps=10, alpha=2/255):
        """
        Generate a dataset with non-robust features using adversarial perturbations.
        :param epsilon: Maximum perturbation size
        :param steps: Number of attack steps
        :param alpha: Step size for iterative attack
        """
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load a standard (non-robust) model
        self.model = ModelLoader(model_name="Standard").load_model().to(self.device).eval()

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

            # FGSM or PGD update step
            perturbation = self.alpha * perturbed_image.grad.sign()
            perturbed_image = perturbed_image + perturbation
            perturbed_image = torch.clamp(perturbed_image, min=image - self.epsilon, max=image + self.epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure valid pixel range
            perturbed_image = perturbed_image.detach().requires_grad_(True)

        return perturbed_image.detach()

    def process_dataset(self):
        """
        Process the entire CIFAR-10 dataset, generating non-robust adversarial examples.
        """
        batch_size = 8  # Adjust if needed
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        non_robust_images, labels = [], []

        for i, (image, label) in enumerate(dataloader):
            print(f"Processing image {i}/{len(dataloader.dataset)}")

            # Generate adversarial examples for each image
            adversarial_image = self.generate_adversarial_example(image, label)
            non_robust_images.append(adversarial_image.cpu())
            labels.append(label)

        # Save dataset
        torch.save((non_robust_images, labels), "results/datasets/nonrobust_cifar10.pth")
        print("Non-robust dataset saved!")

if __name__ == "__main__":
    generator = NonRobustDatasetGenerator()
    generator.process_dataset()
