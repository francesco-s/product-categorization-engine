import os
import logging
from typing import Dict, List, Optional
import torch
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Product Category Classifier API",
    description="API for predicting product categories based on name and brand",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response models
class ProductRequest(BaseModel):
    name: str
    brand: str


class PredictionResponse(BaseModel):
    category: str
    confidence: float
    all_categories: Optional[Dict[str, float]] = None


# DistilBERT classifier model
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


# Import text preprocessing function
from src.utils.text_utils import preprocess_text


# Model singleton for efficient loading
class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_path="./data/models/checkpoints/best_model.pt"):
        if cls._instance is None:
            cls._instance = cls._load_model(model_path)
        return cls._instance

    @classmethod
    def _load_model(cls, model_path):
        try:
            # Load model checkpoint
            checkpoint = torch.load(
                model_path,
                map_location=torch.device('cpu')
            )

            # Get model configuration from checkpoint
            model_config = checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 0)
            model_name = model_config.get('model_name', 'distilbert-base-uncased')
            model_type = model_config.get('model_type', 'distilbert')  # Default to distilbert if not specified

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

            # Detect model architecture based on state dict keys
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

            # Load weights with strict=False to ignore mismatches
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("Model weights loaded successfully")
            except Exception as e:
                logger.warning(f"Issue when loading model weights: {e}")
                logger.info("Attempting to continue with partial model loading")

            # Load id to label mapping
            id_to_category = checkpoint.get('id_to_category', {})

            # If id_to_category is not a dict or is None, initialize it as an empty dict
            if not isinstance(id_to_category, dict):
                logger.warning(f"id_to_category is not a dictionary, initializing as empty dict")
                id_to_category = {}

            # If still empty and we have num_classes, create default mapping
            if not id_to_category and num_classes > 0:
                logger.warning("Creating default category mapping")
                id_to_category = {str(i): f"Category {i}" for i in range(num_classes)}

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            logger.info(f"Model loaded successfully from {model_path}")

            return {
                'model': model,
                'tokenizer': tokenizer,
                'id_to_category': id_to_category
            }
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")


# Dependency to get model
async def get_model():
    try:
        return ModelSingleton.get_instance()
    except Exception as e:
        logger.error(f"Error getting model instance: {e}")
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(product: ProductRequest, model_data=Depends(get_model)):
    try:
        # Get model components
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        id_to_category = model_data['id_to_category']

        # Preprocess input
        name = product.name
        brand = product.brand
        combined_text = f"{name} {brand}"
        preprocessed_text = preprocess_text(combined_text)

        # Tokenize
        tokens = tokenizer(
            preprocessed_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(
                tokens["input_ids"],
                tokens["attention_mask"]
            )
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, dim=1)

            # Convert to Python types
            predicted_class_id = predicted_class.item()
            confidence_value = confidence.item()

            # Get category name from ID
            predicted_category = id_to_category.get(
                str(predicted_class_id),
                f"Category {predicted_class_id}"
            )

            # Get all category probabilities if available
            all_categories = None
            if len(id_to_category) > 0:
                all_probs = probabilities[0].tolist()
                all_categories = {
                    id_to_category.get(str(i), f"Category {i}"): prob
                    for i, prob in enumerate(all_probs)
                }

        # Return response
        return PredictionResponse(
            category=predicted_category,
            confidence=confidence_value,
            all_categories=all_categories
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# Batch prediction endpoint
@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(products: List[ProductRequest], model_data=Depends(get_model)):
    # Implementation similar to the single prediction endpoint
    # but processes a batch of products
    results = []
    for product in products:
        try:
            result = await predict(product, model_data)
            results.append(result)
        except HTTPException as e:
            # Handle errors individually for each product
            results.append(PredictionResponse(
                category="Error",
                confidence=0.0,
                all_categories={"error": str(e.detail)}
            ))
    return results


# Main function to run the API
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)