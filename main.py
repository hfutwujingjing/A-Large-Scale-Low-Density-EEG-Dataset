import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    HBN_PRETRAIN_EPOCHS, HBN_PRETRAIN_LR, HBN_PRETRAIN_BATCH_SIZE,
    HBN_DISTILL_EPOCHS, HBN_DISTILL_LR, HBN_DISTILL_BATCH_SIZE,
    MODMA_TEACHER_EPOCHS, MODMA_TEACHER_LR, MODMA_TEACHER_BATCH_SIZE,
    MODMA_DISTILL_EPOCHS, MODMA_DISTILL_LR, MODMA_DISTILL_BATCH_SIZE,
    EPOCHS, BATCH_SIZE, LR, DEVICE, NUM_CLASSES,
    N_CHANNELS, N_CHANNELS_128, SEQ_LEN, PATCH_SIZE, D_MODEL, NHEAD,
    NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, NORM_FIRST,
    FREEZE_GENERAL, FREEZE_CLASS
)
from data_loader import (
    get_hbn_pretrain_loader, get_hbn_distill_loader,
    get_modma_teacher_dataloaders, get_modma_distill_loader,
    get_own_dataloaders
)
from model import EEGEncoder, PretrainModel, ClassifierModel, DistillModel, DualBranchModel
from train_hbn import pretrain as hbn_pretrain
from distill import distill
from train_modma import train_modma_teacher
from train import train
from test import test_model


def main():
    print(f"Device: {DEVICE}")

    kw_128 = dict(n_channels=N_CHANNELS_128, seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
                  d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                  dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, norm_first=NORM_FIRST)
    kw_3   = dict(n_channels=N_CHANNELS, seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
                  d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                  dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, norm_first=NORM_FIRST)

    # ===== 1a: HBN Teacher Pretrain =====
    print("\n=== 1a: HBN Teacher (128ch Reconstruction) ===")
    hbn_enc_128 = EEGEncoder(**kw_128)
    pm = PretrainModel(hbn_enc_128, N_CHANNELS_128, SEQ_LEN, PATCH_SIZE, D_MODEL)
    pl = get_hbn_pretrain_loader(batch_size=HBN_PRETRAIN_BATCH_SIZE)
    hbn_enc_128 = hbn_pretrain(pm, pl, HBN_PRETRAIN_EPOCHS, HBN_PRETRAIN_LR, DEVICE,
                               "./pth/hbn_teacher_128.pth")
    print()

    # ===== 1b: HBN Distill =====
    print("=== 1b: HBN Distill (128ch -> fp3) ===")
    hbn_enc_3 = EEGEncoder(**kw_3)
    dm = DistillModel(hbn_enc_128, hbn_enc_3)
    dl = get_hbn_distill_loader(batch_size=HBN_DISTILL_BATCH_SIZE)
    hbn_enc_3 = distill(dm, dl, HBN_DISTILL_EPOCHS, HBN_DISTILL_LR, DEVICE,
                        "./pth/hbn_student_3.pth")
    print()

    # ===== 2a: MODMA Teacher =====
    print("=== 2a: MODMA Teacher (128ch Classification) ===")
    modma_enc_128 = EEGEncoder(**kw_128)
    tl, vl_t, n_cls = get_modma_teacher_dataloaders(batch_size=MODMA_TEACHER_BATCH_SIZE)
    cm = ClassifierModel(modma_enc_128, n_cls, D_MODEL)
    modma_enc_128 = train_modma_teacher(cm, tl, vl_t, MODMA_TEACHER_EPOCHS,
                                        MODMA_TEACHER_LR, DEVICE, "./pth/modma_teacher_128.pth")
    print()

    # ===== 2b: MODMA Distill =====
    print("=== 2b: MODMA Distill (128ch -> 3ch) ===")
    modma_enc_3 = EEGEncoder(**kw_3)
    dm2 = DistillModel(modma_enc_128, modma_enc_3)
    dl2 = get_modma_distill_loader(batch_size=MODMA_DISTILL_BATCH_SIZE)
    modma_enc_3 = distill(dm2, dl2, MODMA_DISTILL_EPOCHS, MODMA_DISTILL_LR, DEVICE,
                          "./pth/modma_student_3.pth")
    print()

    # ===== 3: OWN Fine-tune =====
    print("=== 3: OWN Fine-tune (HBN general + MODMA class) ===")
    tr_loader, va_loader, te_loader = get_own_dataloaders(batch_size=BATCH_SIZE)

    model = DualBranchModel(
        encoder_general=hbn_enc_3,
        encoder_class=modma_enc_3,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL
    )

    if FREEZE_GENERAL:
        model.set_general_freeze(True)
        print("  HBN general encoder FREEZED")
    if FREEZE_CLASS:
        model.set_class_freeze(True)
        print("  MODMA class encoder FREEZED")

    model = train(model, tr_loader, va_loader, EPOCHS, LR, DEVICE, FREEZE_GENERAL, FREEZE_CLASS)
    test_model(model, te_loader, DEVICE)


if __name__ == "__main__":
    main()