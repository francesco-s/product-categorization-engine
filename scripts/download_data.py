# scripts/download_data.py

import os
import sys
import argparse
import logging
import pandas as pd
import kagglehub

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('download_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download Amazon products dataset')

    parser.add_argument('--config', type=str, default='configs/training.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the dataset')

    return parser.parse_args()


def main():
    """Main function to download the dataset."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Get output directory
    output_dir = args.output_dir or config.get('data', 'raw_dir')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading Amazon products dataset to {output_dir}")

    try:
        # Download the dataset
        path_to_amazon_product = kagglehub.dataset_download("lokeshparab/amazon-products-dataset")
        logger.info(f"Dataset downloaded to {path_to_amazon_product}")

        # Load the dataset
        amazon_product_df = pd.read_csv(f"{path_to_amazon_product}/Amazon-Products.csv")
        logger.info(f"Loaded dataset with {len(amazon_product_df)} rows")

        # Save to output directory
        output_path = os.path.join(output_dir, "Amazon-Products.csv")
        amazon_product_df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")

        # Print some dataset information
        logger.info(f"Dataset columns: {amazon_product_df.columns.tolist()}")
        logger.info(f"Dataset info:\n{amazon_product_df.info()}")
        logger.info(f"Sample data:\n{amazon_product_df.head()}")

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)

    logger.info("Dataset download completed successfully!")


if __name__ == "__main__":
    main()
