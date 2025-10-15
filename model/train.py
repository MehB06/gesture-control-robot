import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
import logging
from tqdm import tqdm

from .cnn import get_model
from .loader import DataProcessor
from .config import (
    IN_CHANNELS, NUM_CLASSES, BASE_CHANNELS, DROPOUT,
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, LABEL_SMOOTHING,
    USE_SCHEDULER, MIN_LR,
    LAST_CHECKPOINT, BEST_CHECKPOINT
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, save_dir: Path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = build_model(self.model)
        self.criterion = build_criterion()
        self.optimizer = build_optimizer(self.model)
        self.scheduler = build_scheduler(self.optimizer)

        self.save_dir = save_dir
        self.train_loader, self.val_loader, self.test_loader = DataProcessor().getDataLoaders()
        
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Testing')
            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / total:.2f}%'
                })
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100.0 * correct / total
        
        return test_loss, test_acc
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history
        }
        
        checkpoint_path = self.save_dir / LAST_CHECKPOINT
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / BEST_CHECKPOINT
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f'Loaded checkpoint from {checkpoint_path}')
        return checkpoint
    
    def train(self):
        logger.info(f'Starting training for {EPOCHS} epochs')
        logger.info(f'Device: {self.device}')
        logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        for epoch in range(EPOCHS):
            logger.info(f'\nEpoch [{epoch + 1}/{EPOCHS}]')
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, val_loss, val_acc, is_best)
        
        logger.info(f'\nTraining completed!')
        logger.info(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        
        return self.history


def build_model(device: torch.device) -> nn.Module:
    model = get_model(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        dropout=DROPOUT
    )
    return model.to(device)


def build_criterion() -> nn.Module:
    return nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    if not USE_SCHEDULER:
        return None
    
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LR
    )