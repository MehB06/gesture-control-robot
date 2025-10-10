import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple
import logging
from tqdm import tqdm

from cnn import get_model
from loader import trainDataLoader, validationDataLoader, testDataLoader  # Assuming these are directly available
from config import BATCH_SIZE, EPOCHS, TRAIN_PERCENT, TEST_PERCENT, VALIDATE_PERCENT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_dir: Path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
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
        
        checkpoint_path = self.save_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_checkpoint.pth'
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

    def train(self, num_epochs: int):
        logger.info(f'Starting training for {num_epochs} epochs')
        logger.info(f'Device: {self.device}')
        logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        for epoch in range(num_epochs):
            logger.info(f'\nEpoch [{epoch + 1}/{num_epochs}]')
            
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
        in_channels=3,
        num_classes=29,
        base_channels=32,
        dropout=0.3
    )
    return model.to(device)


def build_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )


def build_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6
    )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    save_dir = Path('checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader = trainDataLoader
    val_loader = validationDataLoader
    test_loader = testDataLoader
    
    model = build_model(device)
    criterion = build_criterion()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )
    
    num_epochs = 50
    trainer.train(num_epochs)
    
    logger.info('\nEvaluating on test set...')
    best_checkpoint_path = save_dir / 'best_checkpoint.pth'
    if best_checkpoint_path.exists():
        trainer.load_checkpoint(best_checkpoint_path)
    
    test_loss, test_acc = trainer.test()
    logger.info(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()