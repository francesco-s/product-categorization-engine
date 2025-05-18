# src/training/trainer.py
import os
import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class to handle model training and evaluation
    """
    def __init__(
        self, 
        model, 
        train_dataset, 
        val_dataset, 
        test_dataset=None,
        optimizer=None,
        criterion=None,
        lr_scheduler=None,
        batch_size=32,
        num_epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="./checkpoints",
        early_stopping_patience=3
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            optimizer: PyTorch optimizer (if None, will create AdamW)
            criterion: Loss function (if None, will create CrossEntropyLoss)
            lr_scheduler: Learning rate scheduler (optional)
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            device: Device to use for training (cuda or cpu)
            checkpoint_dir: Directory to save model checkpoints
            early_stopping_patience: Number of epochs to wait before early stopping
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
        else:
            self.test_loader = None
        
        # Set up optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=2e-5, 
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Set up loss function if not provided
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        # Set up learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        # Move model to device
        self.model.to(device)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }

    def train(self):
        """
        Train the model for the specified number of epochs.

        Returns:
            dict: Training history
        """
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Training set size: {len(self.train_dataset)}")
        logger.info(f"Validation set size: {len(self.val_dataset)}")

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()

            # Training step
            train_loss = self._train_epoch(epoch)

            # Validation step
            val_loss, val_metrics = self._validate(epoch)

            # Update learning rate if scheduler exists
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)

            # Save training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])

            # Early stopping and model checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, val_metrics, is_best=True)
                logger.info("✅ New best model saved!")
            else:
                self.epochs_without_improvement += 1
                self._save_checkpoint(epoch, val_loss, val_metrics, is_best=False)

                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
                        f"Val F1: {val_metrics['f1']:.4f}")

        # Final evaluation on test set if available
        if self.test_loader:
            self._load_best_model()
            test_loss, test_metrics = self.evaluate(self.test_loader)  # Changed this line
            logger.info(f"Test Results - "
                        f"Accuracy: {test_metrics['accuracy']:.4f} | "  # Fixed this line
                        f"F1 Score: {test_metrics['f1']:.4f} | "  # Fixed this line
                        f"Precision: {test_metrics['precision']:.4f} | "  # Fixed this line
                        f"Recall: {test_metrics['recall']:.4f}")  # Fixed this line

        return self.training_history
    
    def _train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}/{self.num_epochs} [Train]",
            leave=False
        )
        
        for batch in progress_bar:
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
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def _validate(self, epoch):
        """
        Validate the model on the validation set.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (average_validation_loss, validation_metrics)
        """
        return self.evaluate(
            self.val_loader, 
            desc=f"Epoch {epoch}/{self.num_epochs} [Val]"
        )
    
    def evaluate(self, data_loader, desc="Evaluation"):
        """
        Evaluate the model on a given data loader.
        
        Args:
            data_loader: DataLoader to evaluate on
            desc: Description for the progress bar
            
        Returns:
            tuple: (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=desc, leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Update tracking variables
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, epoch, val_loss, val_metrics, is_best=False):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # If this is the best model, save a copy
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
    
    def _load_best_model(self):
        """
        Load the best model from the checkpoint directory.
        """
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
        else:
            logger.warning(f"Best model checkpoint not found at {best_model_path}")
