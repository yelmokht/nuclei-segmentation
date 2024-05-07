SOURCE_URL = 'https://drive.google.com/file/d/16gp8kPxMFrSDiLjgw2eaZmtrDccOjKKR/view?usp=sharing'
MODEL_URL = 'https://drive.google.com/drive/folders/1u1wopGBqXz7Qn5MnICORHx81nz30mZhL?usp=sharing'
ROOT_PATH = './'
DATA_PATH = ROOT_PATH + 'data/'
SOURCE_PATH = DATA_PATH + 'data-science-bowl-2018.zip'
UNZIPPED_PATH = DATA_PATH + 'unzipped/'
DESTINATION_PATH = DSB_PATH = DATA_PATH + 'dsb-2018/'
TRAIN_PATH = DSB_PATH + 'stage1_train/'
TEST_1_PATH = DSB_PATH + 'stage1_test/'
TEST_2_PATH = DSB_PATH + 'stage2_test/'
STAGE_1_PATH = DSB_PATH + 'stage1_test/'
STAGE_2_PATH = DSB_PATH + 'stage2_test/'
STAGE_1_SOLUTION_PATH = DSB_PATH + 'stage1_solution/stage1_solution.csv'
STAGE_2_SOLUTION_PATH = DSB_PATH + 'stage2_solution/stage2_solution.csv'
IMAGE_HEIGHT = MASK_HEIGHT = 256
IMAGE_WIDTH = MASK_WIDTH = 256
IMAGE_NUM_CHANNELS = 3
MASK_NUM_CHANNELS = 1
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_NUM_CHANNELS)
MASK_SHAPE = (MASK_HEIGHT, MASK_WIDTH, MASK_NUM_CHANNELS)
INPUT_SHAPE = IMAGE_SHAPE
SEED = 42
MODELS_PATH = ROOT_PATH + 'models/'
MODEL_PATH = MODELS_PATH + 'model/'
MODEL_FORMAT = 'model.h5'
HISTORY_FORMAT = 'history.csv'
SUBMISSION_PATH = 'submission.csv'
SOLUTION_PATH = 'solution.csv'