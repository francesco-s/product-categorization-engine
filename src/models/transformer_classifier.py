import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ProductCategoryClassifier(nn.Module):
    """
    Transformer-based model for product category classification
    """
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pre-trained transformer model
            num_classes: Number of category classes
            dropout_rate: Dropout rate for regularization
        """
        super(ProductCategoryClassifier, self).__init__()
        
        # Load pre-trained transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            logits: Unnormalized prediction scores
        """
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
    
    def predict(self, input_ids, attention_mask):
        """
        Get predictions (for inference).
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            predictions: Class predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions


# Factory class for model creation
class ModelFactory:
    """
    Factory class for creating different model variants
    """
    @staticmethod
    def create_model(model_type, num_classes, **kwargs):
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create
            num_classes: Number of category classes
            **kwargs: Additional model parameters
            
        Returns:
            model: Instantiated model
        """
        if model_type == "bert":
            return ProductCategoryClassifier("bert-base-uncased", num_classes, **kwargs)
        elif model_type == "roberta":
            return ProductCategoryClassifier("roberta-base", num_classes, **kwargs)
        elif model_type == "distilbert":
            return ProductCategoryClassifier("distilbert-base-uncased", num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")