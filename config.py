import torch

HBN_DIR = "/16T/wxp/dataset/OSF_MODMA_Ours/HBN_split"
MODMA_DIR = "/16T/wxp/dataset/OSF_MODMA_Ours/MODMA_split"

TRAIN_DIRS = [
    "/16T/wxp/dataset/OSF_MODMA_Ours/Ours_split/Round4_train",
    "/16T/wxp/dataset/OSF_MODMA_Ours/Ours_split/Guangyuan_train",
]
TEST_DIRS = [
    "/16T/wxp/dataset/OSF_MODMA_Ours/Ours_split/Round4_eval",
    "/16T/wxp/dataset/OSF_MODMA_Ours/Ours_split/Guangyuan_eval",
]

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Transformer
N_CHANNELS = 3
N_CHANNELS_128 = 128
SEQ_LEN = 1000
PATCH_SIZE = 50
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.2
NORM_FIRST = True

# Stage 1a: HBN teacher pretrain (reconstruction)
HBN_PRETRAIN_EPOCHS = 50
HBN_PRETRAIN_LR = 1e-4
HBN_PRETRAIN_BATCH_SIZE = 64

# Stage 1b: HBN distill (128ch -> fp3)
HBN_DISTILL_EPOCHS = 50
HBN_DISTILL_LR = 5e-5
HBN_DISTILL_BATCH_SIZE = 128

# Stage 2a: MODMA teacher (classification)
MODMA_TEACHER_EPOCHS = 50
MODMA_TEACHER_LR = 1e-4
MODMA_TEACHER_BATCH_SIZE = 64

# Stage 2b: MODMA distill (128ch -> 3ch)
MODMA_DISTILL_EPOCHS = 50
MODMA_DISTILL_LR = 5e-5
MODMA_DISTILL_BATCH_SIZE = 128

# Stage 3: OWN fine-tune
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-4
FREEZE_GENERAL = False
FREEZE_CLASS = False

NUM_CLASSES = 3