import torch
from pathlib import Path
import logging

from config import SAVE_DIR, BEST_CHECKPOINT
from loader import trainDataLoader, validationDataLoader, testDataLoader
from train import (
    Trainer,
    build_model,
    build_criterion,
    build_optimizer,
    build_scheduler
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Make global variable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Move to train.py (redundent code)
    model = build_model(device)
    criterion = build_criterion()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    trainer = Trainer(
        model=model,
        train_loader=trainDataLoader,
        val_loader=validationDataLoader,
        test_loader=testDataLoader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir
    )

    trainer.train()

    logger.info('\nEvaluating on test set...')
    best_checkpoint_path = save_dir / BEST_CHECKPOINT
    if best_checkpoint_path.exists():
        trainer.load_checkpoint(best_checkpoint_path)

    test_loss, test_acc = trainer.test()
    logger.info(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
