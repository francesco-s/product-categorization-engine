import os
import sys
import argparse
import logging
import json
import yaml
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a trained product categorization model')

    parser.add_argument('--model_path', type=str, default='data/models/best_model.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--test_file', type=str, default='data/processed/test.csv',
                        help='Path to the test data CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--config', type=str, default='configs/evaluation.yaml',
                        help='Path to evaluation configuration file')

    return parser.parse_args()


class DistilBERTClassifier(torch.nn.Module):
    """
    DistilBERT-based classifier for product categories.
    """

    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        """Initialize the model."""
        super(DistilBERTClassifier, self).__init__()

        # Load pre-trained DistilBERT model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        # Classification head
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model."""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use the [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and get logits
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def load_trained_model(model_path):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path: Path to the model checkpoint

    Returns:
        model: The loaded model
        model_config: Model configuration dictionary
        tokenizer: Tokenizer for the model
        id_to_category: Mapping from category IDs to category names
    """
    logger.info(f"Loading model from {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        model_type = model_config.get('model_type', 'distilbert')  # Default to distilbert if not specified
        model_name = model_config.get('model_name', 'distilbert-base-uncased')
        num_classes = model_config.get('num_classes', 0)

        # If num_classes is still 0, try to get it from id_to_category
        if num_classes == 0 and 'id_to_category' in checkpoint:
            num_classes = len(checkpoint['id_to_category'])
            logger.info(f"Detected {num_classes} classes from id_to_category mapping")

        # Also check the classifier size in model_state_dict
        if num_classes == 0 and 'model_state_dict' in checkpoint:
            for key in checkpoint['model_state_dict']:
                if 'classifier.bias' in key:
                    bias_shape = checkpoint['model_state_dict'][key].shape
                    num_classes = bias_shape[0]
                    logger.info(f"Detected {num_classes} classes from classifier.bias shape")
                    break

        logger.info(f"Model type: {model_type}, Model name: {model_name}, Num classes: {num_classes}")

        # Create appropriate model based on the architecture
        if "transformer.transformer.layer" in str(checkpoint['model_state_dict'].keys()):
            logger.info("Detected DistilBERT architecture from checkpoint")
            model = DistilBERTClassifier(
                model_name=model_name,
                num_classes=num_classes
            )
            model_type = "distilbert"
        else:
            # Use the specified model type (default to distilbert)
            if model_type.lower() == "distilbert":
                model = DistilBERTClassifier(
                    model_name=model_name,
                    num_classes=num_classes
                )
            else:
                # Import only if needed
                from src.models.transformer_classifier import ProductCategoryClassifier
                model = ProductCategoryClassifier(
                    model_name=model_name,
                    num_classes=num_classes
                )

        # Load model weights with strict=False to ignore mismatches
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Get category mapping
        id_to_category = checkpoint.get('id_to_category', {})

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, model_config, tokenizer, id_to_category

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_test_data(test_file):
    """
    Load test data from a CSV file.

    Args:
        test_file: Path to the test data CSV file

    Returns:
        test_df: Test data DataFrame
    """
    logger.info(f"Loading test data from {test_file}")

    try:
        test_df = pd.read_csv(test_file)
        logger.info(f"Loaded test data with {len(test_df)} rows")
        return test_df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


class ProductDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for product categorization.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """Initialize dataset."""
        # Convert to list if Series to avoid indexing issues
        if isinstance(texts, pd.Series):
            self.texts = texts.values.tolist()
        else:
            self.texts = texts

        if isinstance(labels, pd.Series):
            self.labels = labels.values.tolist()
        else:
            self.labels = labels

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Get dataset length."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a dataset item."""
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract tensors and squeeze batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        return item


def evaluate_model(model, test_df, tokenizer, id_to_category, batch_size=32, device='cpu'):
    """
    Evaluate the model on test data.

    Args:
        model: Trained model
        test_df: Test data DataFrame
        tokenizer: Tokenizer
        id_to_category: Mapping from category IDs to category names
        batch_size: Batch size for evaluation
        device: Device to use for evaluation

    Returns:
        results: Dictionary of evaluation results
    """
    logger.info(f"Evaluating model on {len(test_df)} test samples")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    from torch.utils.data import DataLoader

    # Check if required columns exist
    if 'combined_text' not in test_df.columns:
        logger.warning("'combined_text' column not found in test data, creating from name and brand")
        if 'name' in test_df.columns and 'brand' in test_df.columns:
            test_df['combined_text'] = test_df['name'] + ' ' + test_df['brand']
        elif 'name' in test_df.columns:
            test_df['combined_text'] = test_df['name']
        else:
            raise ValueError("Required columns not found in test data")

    if 'category_id' not in test_df.columns:
        logger.warning("'category_id' column not found in test data, using 'label' if available")
        if 'label' in test_df.columns:
            test_df['category_id'] = test_df['label']
        else:
            raise ValueError("No category label column found in test data")

    # Create dataset
    test_dataset = ProductDataset(
        texts=test_df['combined_text'],
        labels=test_df['category_id'],
        tokenizer=tokenizer,
        max_length=128
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            # Convert tensors to lists immediately
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Convert IDs to category names
    pred_categories = [id_to_category.get(str(idx), f"Unknown_{idx}") for idx in all_preds]
    true_categories = [id_to_category.get(str(idx), f"Unknown_{idx}") for idx in all_labels]

    # Calculate metrics
    # Check if id_to_category is empty or None
    if not id_to_category:
        logger.warning("id_to_category dictionary is empty, generating default class names")
        # Generate default class names based on unique labels in predictions
        unique_labels = sorted(set(all_labels).union(set(all_preds)))
        target_names = [f"Class_{i}" for i in unique_labels]
        # Also update id_to_category for consistency
        id_to_category = {str(i): f"Class_{i}" for i in unique_labels}
    else:
        # Generate target_names from num_classes rather than len(id_to_category)
        num_classes = max(max(all_labels), max(all_preds)) + 1
        target_names = [id_to_category.get(str(i), f"Class_{i}") for i in range(num_classes)]

    # Generate classification report with proper target_names
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(target_names))),  # Ensure labels match target_names
        target_names=target_names,
        output_dict=True,
        zero_division=0  # Avoid warnings for undefined precision
    )

    # Get confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Convert NumPy arrays to lists if needed (handle case where they might already be lists)
    all_preds_list = all_preds.tolist() if hasattr(all_preds, 'tolist') else all_preds
    all_labels_list = all_labels.tolist() if hasattr(all_labels, 'tolist') else all_labels
    cm_list = cm.tolist() if hasattr(cm, 'tolist') else cm

    # Combine results
    results = {
        'classification_report': report,
        'confusion_matrix': cm_list,
        'predictions': all_preds_list,
        'true_labels': all_labels_list,
        'pred_categories': pred_categories,
        'true_categories': true_categories
    }

    # Print summary
    accuracy = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    logger.info(f"Evaluation results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Macro F1: {macro_f1:.4f}")
    logger.info(f"  Weighted F1: {weighted_f1:.4f}")

    return results


def analyze_misclassifications(results, test_df, max_samples=20):
    """
    Analyze misclassifications.

    Args:
        results: Evaluation results
        test_df: Test data DataFrame
        max_samples: Maximum number of misclassification examples to include

    Returns:
        misclass_analysis: Misclassification analysis
    """
    logger.info("Analyzing misclassifications")

    # Get indices of misclassifications
    predictions = np.array(results['predictions'])
    true_labels = np.array(results['true_labels'])
    misclass_indices = np.where(predictions != true_labels)[0]

    # Get misclassification examples
    misclass_examples = []

    for idx in misclass_indices[:max_samples]:
        if 'name' in test_df.columns and 'brand' in test_df.columns:
            example = {
                'text': test_df.iloc[idx]['name'],
                'brand': test_df.iloc[idx]['brand'],
                'combined_text': test_df.iloc[idx]['combined_text'],
                'true_category': results['true_categories'][idx],
                'pred_category': results['pred_categories'][idx]
            }
        else:
            example = {
                'combined_text': test_df.iloc[idx]['combined_text'],
                'true_category': results['true_categories'][idx],
                'pred_category': results['pred_categories'][idx]
            }
        misclass_examples.append(example)

    # Get most common misclassifications
    misclass_pairs = []
    for idx in misclass_indices:
        true_cat = results['true_categories'][idx]
        pred_cat = results['pred_categories'][idx]
        misclass_pairs.append((true_cat, pred_cat))

    from collections import Counter
    common_misclass = Counter(misclass_pairs).most_common(10)

    # Create analysis
    misclass_analysis = {
        'num_misclassifications': len(misclass_indices),
        'misclassification_rate': len(misclass_indices) / len(predictions),
        'examples': misclass_examples,
        'common_misclassifications': [
            {
                'true_category': pair[0][0],
                'pred_category': pair[0][1],
                'count': pair[1]
            }
            for pair in common_misclass
        ]
    }

    logger.info(
        f"Found {len(misclass_indices)} misclassifications ({misclass_analysis['misclassification_rate']:.4f} error rate)")

    return misclass_analysis


def plot_confusion_matrix(cm, classes, output_path, n_classes=20):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        classes: Class names
        output_path: Path to save plot
        n_classes: Number of top classes to plot
    """
    # Get top classes by frequency
    class_counts = np.sum(cm, axis=1)
    top_indices = np.argsort(class_counts)[-n_classes:]

    # Filter confusion matrix for top classes
    cm_top = cm[top_indices, :][:, top_indices]
    classes_top = [classes[i] for i in top_indices]

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes_top, yticklabels=classes_top)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Top Classes)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Confusion matrix plot saved to {output_path}")


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    model, model_config, tokenizer, id_to_category = load_trained_model(args.model_path)

    # Load test data
    test_df = load_test_data(args.test_file)

    # Evaluate model
    results = evaluate_model(
        model=model,
        test_df=test_df,
        tokenizer=tokenizer,
        id_to_category=id_to_category,
        batch_size=args.batch_size,
        device=device
    )

    # Analyze misclassifications
    misclass_analysis = analyze_misclassifications(
        results=results,
        test_df=test_df,
        max_samples=20
    )

    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'metrics': results['classification_report'],
                'misclassification_analysis': misclass_analysis
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    logger.info(f"Evaluation results saved to {results_path}")

    # Plot confusion matrix if class names are available
    if id_to_category:
        class_names = [id_to_category.get(str(i), f"Class_{i}") for i in range(len(id_to_category))]
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            cm=np.array(results['confusion_matrix']),
            classes=class_names,
            output_path=cm_path,
            n_classes=20
        )

    # Print summary
    logger.info("Evaluation completed successfully!")
    logger.info(f"  Model: {model_config.get('model_type', 'unknown')}")
    logger.info(f"  Number of classes: {len(id_to_category)}")
    logger.info(f"  Test samples: {len(test_df)}")
    logger.info(f"  Accuracy: {results['classification_report']['accuracy']:.4f}")
    logger.info(f"  Macro F1: {results['classification_report']['macro avg']['f1-score']:.4f}")
    logger.info(f"  Weighted F1: {results['classification_report']['weighted avg']['f1-score']:.4f}")

    # List top misclassifications
    if misclass_analysis['common_misclassifications']:
        logger.info("Top 5 most common misclassifications:")
        for i, misclass in enumerate(misclass_analysis['common_misclassifications'][:5]):
            logger.info(
                f"  {i + 1}. {misclass['true_category']} -> {misclass['pred_category']} ({misclass['count']} occurrences)")


if __name__ == "__main__":
    main()