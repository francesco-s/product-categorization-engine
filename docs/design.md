# Design Documentation: Product Categorization

This document outlines the design decisions, architecture, and trade-offs for a product categorization solution.

## 1. Problem Statement

A client needs a machine learning solution that can automatically predict the correct category for a product based on its name and brand. The solution should be:

- Accurate in classifying products across multiple categories
- Accessible via a REST API
- Production-ready with good code practices
- Properly packaged and documented

## 2. Solution Architecture

The solution follows a modular architecture with clear separation of concerns:

```
                             ┌─────────────────┐
                             │                 │
                             │  Text Processor │
                             │                 │
                             └────────┬────────┘
                                      │
                                      ▼
┌─────────────┐            ┌─────────────────┐            ┌─────────────────┐
│             │            │                 │            │                 │
│   Product   │────────────▶     Model      │────────────▶     API         │
│    Data     │            │                 │            │                 │
│             │            └─────────────────┘            └─────────────────┘
└─────────────┘                    ▲                              │
                                   │                              │
                                   │                              ▼
                            ┌──────┴────────┐            ┌─────────────────┐
                            │               │            │                 │
                            │    Trainer    │            │     Client      │
                            │               │            │                 │
                            └───────────────┘            └─────────────────┘
```

### 2.1 Components

1. **Text Processor**: Handles text preprocessing, cleaning, and feature engineering
   - Implements various text cleaning strategies (lowercasing, stop word removal, etc.)
   - Combines product name and brand information
   - Normalizes text for consistent model input

2. **Model**: Implements the transformer-based classifier with fine-tuning
   - Uses pre-trained transformer models (BERT, RoBERTa, DistilBERT)
   - Applies transfer learning through fine-tuning
   - Implements a classification head on top of the transformer

3. **Trainer**: Manages model training, evaluation, and checkpointing 
   - Implements training loop with early stopping
   - Handles model evaluation
   - Manages checkpointing and model persistence

4. **API**: Provides a REST interface for making predictions
   - Implements FastAPI endpoints for single and batch predictions
   - Handles request validation and error handling
   - Manages model loading and inference

5. **Client**: External systems that consume the API

### 2.2 Design Patterns

The solution implements several design patterns to ensure good code organization and maintainability:

#### Factory Pattern
Used for creating different model variants:

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type, num_classes, **kwargs):
        if model_type == "bert":
            return ProductCategoryClassifier("bert-base-uncased", num_classes, **kwargs)
        elif model_type == "roberta":
            return ProductCategoryClassifier("roberta-base", num_classes, **kwargs)
        elif model_type == "distilbert":
            return ProductCategoryClassifier("distilbert-base-uncased", num_classes, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
```

This allows for creating different model variants without changing the client code, following the Open/Closed Principle.

#### Strategy Pattern
For different text processing strategies:

```python
class TextPreprocessor:
    def __init__(self, lowercase=True, remove_stopwords=True, ...):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        # ...
        
    def preprocess(self, text):
        # Apply configured preprocessing strategies
        if self.lowercase:
            text = text.lower()
        # ...
```

This allows configuring different text preprocessing strategies without changing the processor's structure.

#### Repository Pattern
For data access abstraction:

```python
class ProductDataProcessor:
    def process(self, df, name_column, brand_column, ...):
        # Process data and abstract storage details
```

This abstracts the data storage and retrieval details from the rest of the application.

#### Singleton Pattern
For model loading in the API:

```python
class ModelSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path):
        if cls._instance is None:
            cls._instance = cls._load_model(model_path)
        return cls._instance
```

This ensures the model is loaded only once, saving memory and improving API performance.

#### Dependency Injection
For flexible component configuration:

```python
def create_data_loaders(train_df, val_df, test_df, tokenizer, ...):
    # Components are injected rather than created internally
```

This makes the code more testable and flexible.

## 3. Model Selection and Fine-tuning

### 3.1 Transformer-Based Fine-tuning

After evaluating several approaches, we chose a transformer-based fine-tuning solution for product categorization:

#### Fine-tuning vs. Feature Extraction

We chose **fine-tuning** rather than feature extraction:

- **Fine-tuning**: Update all model parameters (both pre-trained transformer weights and new classification head)
- **Feature Extraction**: Freeze pre-trained weights and only train the classification head

Fine-tuning allows the model to adapt its pre-trained language understanding to the specific domain of product descriptions and categories.

Our implementation allows all model parameters to be updated:

```python
# In training script
optimizer = torch.optim.AdamW(
    model.parameters(),  # This includes ALL model parameters
    lr=learning_rate,
    weight_decay=weight_decay
)
```

#### Available Models

We implemented support for three transformer models:

| Model | Type | Parameters | Training Speed | Inference Speed | Relative Accuracy |
|-------|------|------------|----------------|-----------------|-------------------|
| BERT | bert-base-uncased | 110M | Slower | Slower | Baseline (100%) |
| RoBERTa | roberta-base | 125M | Slower | Slower | 100-102% |
| DistilBERT | distilbert-base-uncased | 66M | 60% faster | 60% faster | ~97% |

### 3.2 Model Architecture

```python
class ProductCategoryClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.1):
        super(ProductCategoryClassifier, self).__init__()
        
        # Load pre-trained transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
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
```

The architecture consists of:
1. **Pre-trained transformer encoder**: Processes text inputs to extract contextual representations
2. **Dropout layer**: Prevents overfitting through regularization
3. **Linear classification head**: Maps the [CLS] token representation to category predictions

### 3.3 Fine-tuning Optimizations

Our implementation includes several best practices for fine-tuning transformer models:

1. **Lower Learning Rate**: We use a small learning rate (2e-5) to avoid catastrophic forgetting.
2. **Weight Decay**: Applied through AdamW optimizer (0.01) to prevent overfitting.
3. **Early Stopping**: Training stops when validation loss stops improving.
4. **Gradient Clipping**: To stabilize training with a maximum norm of 1.0.
5. **Dropout**: Applied before the classification head to improve generalization.

## 4. Data Processing Pipeline

The data processing pipeline implements these steps:

### 4.1 Text Cleaning

The `TextPreprocessor` class handles various text cleaning operations:

```python
def preprocess(self, text):
    # Convert to lowercase
    if self.lowercase:
        text = text.lower()
    
    # Remove punctuation
    if self.remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove special characters, tokenize, remove stopwords, lemmatize, etc.
    # ...
    
    return text
```

### 4.2 Feature Engineering

The system combines product name and brand information into a single text input:

```python
df_processed['combined_text'] = (
    df_processed[f'processed_{text_column}'] + ' ' + 
    df_processed[f'processed_{brand_column}']
)
```

This allows the model to consider both pieces of information in making category predictions.

### 4.3 Category Handling

The `ProductDataProcessor` handles category encoding and data splitting:

```python
def _encode_categories(self, categories):
    unique_categories = categories.unique()
    self.category_to_id = {category: idx for idx, category in enumerate(unique_categories)}
    self.id_to_category = {idx: category for category, idx in self.category_to_id.items()}
    self.num_classes = len(self.category_to_id)
```

### 4.4 Dataset Creation

The `ProductDataset` class creates PyTorch datasets for training:

```python
class ProductDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```

## 5. Training Process

The training process follows these steps:

### 5.1 Training Loop

The `Trainer` class implements the training loop:

```python
def train(self):
    for epoch in range(1, self.num_epochs + 1):
        # Training step
        train_loss = self._train_epoch(epoch)
        
        # Validation step
        val_loss, val_metrics = self._validate(epoch)
        
        # Early stopping and checkpointing
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self._save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
```

### 5.2 Training Step

Training individual epochs:

```python
def _train_epoch(self, epoch):
    self.model.train()
    total_loss = 0
    
    for batch in self.train_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(self.train_loader)
```

### 5.3 Evaluation

Evaluating model performance:

```python
def evaluate(self, data_loader):
    self.model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Forward pass
            outputs = self.model(batch['input_ids'], batch['attention_mask'])
            loss = self.criterion(outputs, batch['labels'])
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted')
    }
    
    return total_loss / len(data_loader), metrics
```

## 6. API Design

The REST API is implemented using FastAPI:

### 6.1 API Endpoints

```python
# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(product: ProductRequest, model_data=Depends(get_model)):
    # ... implementation ...

# Batch prediction endpoint
@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(products: List[ProductRequest], model_data=Depends(get_model)):
    # ... implementation ...
```

### 6.2 Model Loading

The API uses a Singleton pattern for efficient model loading:

```python
class ModelSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_path="./data/models/best_model.pt"):
        if cls._instance is None:
            cls._instance = cls._load_model(model_path)
        return cls._instance
```

### 6.3 Request/Response Models

Using Pydantic for request/response validation:

```python
class ProductRequest(BaseModel):
    name: str
    brand: str
    
class PredictionResponse(BaseModel):
    category: str
    confidence: float
    all_categories: Optional[Dict[str, float]] = None
```

## 7. Deployment Strategy

### 7.1 Containerization

The solution is containerized using Docker:

```Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Start the application
CMD ["/app/scripts/start.sh"]
```

### 7.2 Service Orchestration

Using Docker Compose for orchestration:

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: product-categorization-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    restart: unless-stopped

  monitoring:
    image: prom/prometheus:latest
    container_name: product-categorization-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - api
```

### 7.3 Scalability

The solution is designed for horizontal scaling:

- Stateless API design allows deploying multiple instances
- Model loading uses Singleton pattern to avoid redundant loading
- Docker containers enable easy scaling in cloud environments

## 8. Performance Considerations

### 8.1 Model Size vs. Speed Trade-offs

We provide options for different model sizes to balance performance and accuracy:

- **BERT**: Best accuracy, slower inference
- **RoBERTa**: Similar to BERT, potentially better accuracy
- **DistilBERT**: 40% smaller, 60% faster, ~97% of BERT's accuracy

### 8.2 Batch Processing

The API supports batch processing for higher throughput:

```python
@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(products: List[ProductRequest], model_data=Depends(get_model)):
    results = []
    for product in products:
        try:
            result = await predict(product, model_data)
            results.append(result)
        except HTTPException as e:
            results.append(PredictionResponse(
                category="Error",
                confidence=0.0,
                all_categories={"error": str(e.detail)}
            ))
    return results
```

### 8.3 Adaptive Optimization

For resource-constrained environments, we provide options for:

- Smaller models (DistilBERT)
- Reduced batch sizes
- Subset training
- CPU inference when GPU is unavailable

## 9. Future Improvements

Potential improvements for future iterations:

### 9.1 Model Optimization

- **Quantization**: Convert model to int8 precision for faster inference
- **Pruning**: Remove redundant weights for smaller models
- **Knowledge Distillation**: Train smaller custom models from larger ones

### 9.2 Active Learning

- Implement feedback loop for continuous model improvement
- Track prediction confidence to identify uncertain cases
- Request human labeling for uncertain predictions

### 9.3 Multi-Modal Modeling

- Incorporate product image data when available
- Implement multi-modal models combining text and images
- Use pre-trained vision-language models like CLIP

### 9.4 Enhanced Monitoring

- A/B testing framework for comparing model versions
- Drift detection to identify when retraining is needed
- Performance monitoring for prediction latency

## 10. Trade-offs and Decisions

### 10.1 Accuracy vs. Speed

- **Decision**: Prioritize accuracy while providing options for faster inference
- **Rationale**: Category accuracy directly impacts user experience
- **Mitigation**: Provide DistilBERT option for cases where speed is critical

### 10.2 Complexity vs. Maintainability

- **Decision**: Modular design with clear separation of concerns
- **Rationale**: Ensures maintainability despite increased initial complexity
- **Benefit**: Components can be independently updated or replaced

### 10.3 Fine-tuning vs. Feature Extraction

- **Decision**: Implement fine-tuning rather than feature extraction
- **Rationale**: Fine-tuning allows adapting to the specific domain
- **Trade-off**: Higher computational cost but better performance

### 10.4 Framework Selection

- **Decision**: PyTorch for model implementation, FastAPI for API
- **Rationale**: PyTorch provides flexibility for research; FastAPI is fast and modern
- **Alternative Considered**: TensorFlow/Keras with Flask, which would be more integrated but less flexible

## 11. Conclusion

The designed solution provides a robust, production-ready system for product categorization that balances accuracy, performance, and maintainability. The modular architecture allows for future improvements and adaptations, while the current implementation delivers strong performance on the core task.

The choice of fine-tuning transformer models provides superior accuracy compared to traditional ML approaches, while offering flexibility in model selection to balance resource constraints and performance requirements.
