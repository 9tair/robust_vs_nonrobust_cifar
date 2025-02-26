import torch
from robustbench.utils import load_model

class ModelLoader:
    def __init__(self, model_name="Wang2023Better_WRN-28-10", dataset="cifar10", threat_model="L2"):
        """
        Load a robust CIFAR-10 model from RobustBench.
        Available models: https://github.com/RobustBench/robustbench
        """
        self.model_name = model_name
        self.dataset = dataset
        self.threat_model = threat_model

    def load_model(self):
        """Load the pre-trained robust model from RobustBench"""
        model = load_model(model_name=self.model_name, dataset=self.dataset, threat_model=self.threat_model)
        return model.cuda().eval()