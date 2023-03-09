IMG_PATH = "../dataset/auris_seg/images/"
LAB_PATH = "../dataset/auris_seg/labels/"
VIDEO_FRAMES_PATH = "../dataset/auris_seg/all_frames/"

# IMG_PATH = '../../dataset/segmentation/auris_seg/dataset/images/'
# LAB_PATH = '../../dataset/segmentation/auris_seg/dataset/labels/'
# VIDEO_FRAMES_PATH = '../../dataset/segmentation/auris_seg/all_frames/'

FRAME_KEYWORD = 'frame'
FILE_TYPE = '.png'
CLASS_ID_TYPE = 'xx-xx/'
CLASS_ID_CUT = '/'

PRETRAINED_PATH = "../pretrained_models/deeplabV3/"
PRINT_PATH = "results/test_auris.txt"
CONTINUE_FOLDER = './checkpoints/'
CONTINUE_PATH = './checkpoints/continue.pth'
VAL_MODEL_FOLDER = "./val_models/"
SAVE_MASK_PATH = "./ML_preds/"