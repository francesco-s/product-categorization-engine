#!/usr/bin/env python
# scripts/train.py

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.preprocessing import TextPreprocessor, ProductDataProcessor
from src.data.dataset import create_data_loaders
from src.models.transformer_classifier import ModelFactory
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train product categorization model')

    parser.add_argument('--config', type=str, default='configs/training.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='data/raw/Amazon-Products.csv',
                        help='Path to input data file')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Model type (bert, roberta, distilbert)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save model and results')

    return parser.parse_args()


def extract_brand_from_name(df):
    """Extract brand from product name column."""
    logger.info("Extracting brand from product name")
    # Extract first word as brand (adjust regex as needed)
    df['brand'] = df['name'].str.extract(r'^([\w&-]+)')
    return df


def process_categories(df):
    """Process category information."""
    logger.info("Processing category information")
    # If sub_category exists, combine it with main_category
    if 'sub_category' in df.columns:
        df['categories'] = df['main_category'] + ' > ' + df['sub_category']
    else:
        # Otherwise just use main_category
        df['categories'] = df['main_category']
    return df


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if args.model_type:
        config.config['model']['model_type'] = args.model_type
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config.config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config.config['training']['learning_rate'] = args.learning_rate
    if args.output_dir:
        config.config['training']['checkpoint_dir'] = args.output_dir

    # Create output directories
    os.makedirs(config.get('data', 'processed_dir'), exist_ok=True)
    os.makedirs(config.get('data', 'models_dir'), exist_ok=True)
    os.makedirs(config.get('training', 'checkpoint_dir'), exist_ok=True)

    # Load dataset
    data_path = args.data_path or config.get('data', 'amazon_products_file')
    logger.info(f"Loading dataset from {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} rows")

        # Check dataset structure
        logger.info(f"Dataset columns: {df.columns.tolist()}")

        # Extract brand from name if 'brand' column doesn't exist
        if 'brand' not in df.columns:
            df = extract_brand_from_name(df)

        # Process category information
        if 'categories' not in df.columns:
            df = process_categories(df)

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    # Initialize text preprocessor
    text_preprocessor = TextPreprocessor(
        lowercase=config.get('preprocessing', 'lowercase'),
        remove_special_chars=config.get('preprocessing', 'remove_special_chars'),
        remove_stopwords=config.get('preprocessing', 'remove_stopwords'),
        lemmatize=config.get('preprocessing', 'lemmatize')
    )

    # Initialize data processor
    data_processor = ProductDataProcessor(
        text_preprocessor=text_preprocessor,
        test_size=config.get('preprocessing', 'test_size'),
        val_size=config.get('preprocessing', 'val_size'),
        random_state=config.get('preprocessing', 'random_state')
    )

    # Process dataset
    train_df, val_df, test_df = data_processor.process(
        df,
        name_column='name',
        brand_column='brand',
        category_column='categories',
        save_path=config.get('data', 'processed_dir')
    )

    # Get number of classes
    num_classes = data_processor.get_num_classes()
    category_to_id, id_to_category = data_processor.get_category_mappings()

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Example categories: {list(category_to_id.keys())[:5]}")

    # Initialize tokenizer
    model_name = config.get('model', 'model_name')
    logger.info(f"Initializing tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create data loaders
    logger.info("Creating data loaders")
    data_loaders = create_data_loaders(
        train_df,
        val_df,
        test_df,
        tokenizer,
        text_column='combined_text',
        label_column='category_id',
        batch_size=config.get('training', 'batch_size'),
        max_length=config.get('preprocessing', 'max_length')
    )

    # Create model
    model_type = config.get('model', 'model_type')
    logger.info(f"Creating model: {model_type}")
    model = ModelFactory.create_model(
        model_type=model_type,
        num_classes=num_classes,
        dropout_rate=config.get('model', 'dropout_rate')
    )

    # Get learning rate and weight decay as float (fix for the TypeError)
    learning_rate = float(config.get('training', 'learning_rate'))
    weight_decay = float(config.get('training', 'weight_decay'))

    # Create optimizer with explicit float conversion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=data_loaders['train_dataset'],
        val_dataset=data_loaders['val_dataset'],
        test_dataset=data_loaders['test_dataset'],
        optimizer=optimizer,
        lr_scheduler=scheduler,
        batch_size=config.get('training', 'batch_size'),
        num_epochs=config.get('training', 'num_epochs'),
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir=config.get('training', 'checkpoint_dir'),
        early_stopping_patience=config.get('training', 'early_stopping_patience')
    )

    # Train model
    logger.info("Starting training")
    training_history = trainer.train()

    # Save additional model information
    final_model_path = os.path.join(config.get('data', 'models_dir'), "best_model.pt")
    logger.info(f"Saving final model to {final_model_path}")

    # Load best model checkpoint
    best_model_path = os.path.join(config.get('training', 'checkpoint_dir'), "best_model.pt")
    checkpoint = torch.load(best_model_path)

    # Add additional information to checkpoint
    checkpoint['model_config'] = {
        'model_type': model_type,
        'model_name': model_name,
        'num_classes': num_classes,
        'max_length': config.get('preprocessing', 'max_length')
    }
    checkpoint['id_to_category'] = id_to_category
    checkpoint['category_to_id'] = category_to_id

    # Save final model
    torch.save(checkpoint, final_model_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
