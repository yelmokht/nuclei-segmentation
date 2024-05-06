SOURCE_URL = 'https://drive.google.com/file/d/16gp8kPxMFrSDiLjgw2eaZmtrDccOjKKR/view?usp=sharing'
ROOT_PATH = './'
SOURCE_PATH = ROOT_PATH + 'data/' + 'data-science-bowl-2018.zip'
UNZIPPED_PATH = ROOT_PATH + 'data/' + 'unzipped/'
DESTINATION_PATH = DATA_PATH = ROOT_PATH + 'data/' + 'dsb-2018/'
TRAIN_PATH = DATA_PATH + 'stage1_train/'
TEST_1_PATH = DATA_PATH + 'stage1_test/'
TEST_2_PATH = DATA_PATH + 'stage2_test/'
STAGE_1_PATH = DATA_PATH + 'stage1_test/'
STAGE_2_PATH = DATA_PATH + 'stage2_test/'
STAGE_1_SOLUTION_PATH = DATA_PATH + 'stage1_solution/stage1_solution.csv'
STAGE_2_SOLUTION_PATH = DATA_PATH + 'stage2_solution/stage2_solution.csv'
IMAGE_HEIGHT = MASK_HEIGHT = 256
IMAGE_WIDTH = MASK_WIDTH = 256
IMAGE_NUM_CHANNELS = 3
MASK_NUM_CHANNELS = 1
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NUM_CHANNELS)
MASK_SHAPE = (MASK_HEIGHT, MASK_WIDTH, MASK_NUM_CHANNELS)
INPUT_SHAPE = IMAGE_SHAPE
SEED = 42
BATCH_SIZE = 8
EPOCHS = 100
SAVE_MODEL_PATH = f'model_{EPOCHS}.keras'
SAVE_HISTORY_PATH = f'model_{EPOCHS}_history.csv'
LOAD_MODEL_PATH = f'model_{EPOCHS}.keras'
LOAD_HISTORY_PATH = f'model_{EPOCHS}_history.csv'
BEST_MODEL_PATH = 'best_model.keras'
BEST_HISTORY_PATH = 'best_history.csv'
MODELS_PATH = ROOT_PATH + 'models/'
SUBMISSION_PATH = 'submission.csv'
SOLUTION_PATH = 'solution.csv'