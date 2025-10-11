# Model Parameters
IN_CHANNELS = 3
NUM_CLASSES = 29
BASE_CHANNELS = 32
DROPOUT = 0.3

# Training Parameters
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
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