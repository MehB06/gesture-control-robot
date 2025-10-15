from pathlib import Path
import logging

from model.config import SAVE_DIR, BEST_CHECKPOINT
from model.loader import trainDataLoader, validationDataLoader, testDataLoader
from model.train import Trainer

from model.imageCapture import ASLDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / BEST_CHECKPOINT).exists():
        logger.info("Loading ASL Detector UI.")
        stream = ASLDetector(save_dir=save_dir)
    else:
        logger.info("Training new model as saved not found...")
    

        trainer = Trainer(
            train_loader=trainDataLoader,
            val_loader=validationDataLoader,
            test_loader=testDataLoader,
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
