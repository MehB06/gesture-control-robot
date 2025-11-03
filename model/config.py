# Model Parameters
IN_CHANNELS = 1
NUM_CLASSES = 29
BASE_CHANNELS = 32
DROPOUT = 0.3

# Training Parameters
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
USE_SCHEDULER = True
MIN_LR = 1e-6

# Data Loading
NUM_WORKERS = 4

# Data Split
TRAIN_PERCENT = 0.7
VALIDATE_PERCENT = 0.15
TEST_PERCENT = 0.15

# Paths
SAVE_DIR = 'data/saves/'
DATA_DIR = 'data/raw/'

# Checkpoint filenames
LAST_CHECKPOINT = 'last_checkpoint.pth'
BEST_CHECKPOINT = 'best_checkpoint.pth'