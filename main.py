import argparse
from datasets.generate_robust_dataset import RobustDatasetGenerator
from datasets.generate_nonrobust_dataset import NonRobustDatasetGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for dataset generation.")
    parser.add_argument("--generate_robust", action="store_true", help="Generate the robust dataset")
    parser.add_argument("--generate_nonrobust", action="store_true", help="Generate the non-robust dataset")

    args = parser.parse_args()

    if args.generate_robust:
        generator = RobustDatasetGenerator()
        generator.process_dataset()

    if args.generate_nonrobust:
        generator = NonRobustDatasetGenerator()
        generator.process_dataset()
