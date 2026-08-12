import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import HBN_DIR, MODMA_DIR, TRAIN_DIRS, TEST_DIRS


FP3_IDX_HBN = [21, 15, 8]
FP3_IDX_MODMA = [21, 10, 8]

def load_npz_safe(path):
    try:
        return np.load(path, allow_pickle=True)
    except:
        return None


# ==================== HBN ====================

class HBN128Dataset(Dataset):
    def __init__(self):
        self.segments = []
        for fname in ["HBN_rest_segments.npz", "HBN_stim_segments.npz"]:
            data = load_npz_safe(os.path.join(HBN_DIR, fname))
            if data is None:
                continue
            for sub_id in tqdm(data.files, desc=f"HBN teacher {fname}"):
                segs = data[sub_id].item()["segments"].astype(np.float32)
                if segs.shape[1] > 128:
                    segs = segs[:, :128, :]
                if segs.shape[2] == 2000:
                    segs = segs[:, :, ::2]
                self.segments.append(segs)
        self.segments = np.concatenate(self.segments, axis=0) if self.segments else np.array([])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.segments[idx])
        return x, x


class HBNDistillDataset(Dataset):
    def __init__(self):
        self.pairs = []
        for cond in ["rest", "stim"]:
            d128 = load_npz_safe(os.path.join(HBN_DIR, f"HBN_{cond}_segments.npz"))
            dfp3 = load_npz_safe(os.path.join(HBN_DIR, f"HBN_{cond}_fp3_segments.npz"))
            if d128 is None or dfp3 is None:
                continue
            common = sorted(set(d128.files) & set(dfp3.files))
            for sub_id in tqdm(common, desc=f"HBN distill {cond}"):
                s128 = d128[sub_id].item()["segments"].astype(np.float32)
                s3 = dfp3[sub_id].item()["segments"].astype(np.float32)
                if s128.shape[1] > 128:
                    s128 = s128[:, :128, :]
                if s128.shape[2] == 2000:
                    s128 = s128[:, :, ::2]
                if s3.shape[2] == 2000:
                    s3 = s3[:, :, ::2]
                n = min(len(s128), len(s3))
                for i in range(n):
                    self.pairs.append((s128[i], s3[i]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x128, x3 = self.pairs[idx]
        return torch.from_numpy(x128), torch.from_numpy(x3)


# ==================== MODMA ====================

class MODMAClassDataset(Dataset):
    def __init__(self, sub_ids, sub_data, sub_labels):
        self.segments = []
        self.labels = []
        for sid in sub_ids:
            for seg in sub_data[sid]:
                self.segments.append(seg)
                self.labels.append(sub_labels[sid])
        self.segments = np.array(self.segments, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.segments[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def collect_modma_subjects():
    subjects = {}
    for fname in ["MODMA_rest_128ch_segments.npz", "MODMA_erp_128ch_segments.npz"]:
        data = load_npz_safe(os.path.join(MODMA_DIR, fname))
        if data is None:
            continue
        for sub_id in tqdm(data.files, desc=f"MODMA teacher {fname}"):
            item = data[sub_id].item()
            segs = item["segments"].astype(np.float32)
            label = item["label"]
            if segs.shape[1] > 128:
                segs = segs[:, :128, :]
            if sub_id not in subjects:
                subjects[sub_id] = {"segments": [], "label": label}
            for i in range(len(segs)):
                subjects[sub_id]["segments"].append(segs[i])
    return subjects


class MODMADistillDataset(Dataset):
    def __init__(self):
        self.pairs = []
        for cond in ["rest", "erp"]:
            d128 = load_npz_safe(os.path.join(MODMA_DIR, f"MODMA_{cond}_128ch_segments.npz"))
            if d128 is None:
                continue
            for sub_id in tqdm(d128.files, desc=f"MODMA distill {cond}"):
                s128 = d128[sub_id].item()["segments"].astype(np.float32)
                if s128.shape[1] > 128:
                    s128 = s128[:, :128, :]
                s3 = s128[:, FP3_IDX_MODMA, :]
                for i in range(len(s128)):
                    self.pairs.append((s128[i], s3[i]))

        d3ch = load_npz_safe(os.path.join(MODMA_DIR, "MODMA_3ch_segments.npz"))
        if d3ch is not None:
            pool_128 = {}
            for cond in ["rest", "erp"]:
                d128 = load_npz_safe(os.path.join(MODMA_DIR, f"MODMA_{cond}_128ch_segments.npz"))
                if d128 is None:
                    continue
                for sub_id in tqdm(d128.files, desc=f"pool_128 for {cond}"):
                    s128 = d128[sub_id].item()["segments"].astype(np.float32)
                    if s128.shape[1] > 128:
                        s128 = s128[:, :128, :]
                    pool_128.setdefault(sub_id, []).append(s128)
            for sub_id in tqdm(d3ch.files, desc="MODMA distill 3ch"):
                if sub_id not in pool_128:
                    continue
                s3ch = d3ch[sub_id].item()["segments"].astype(np.float32)
                pool = np.concatenate(pool_128[sub_id], axis=0)
                for i in range(len(s3ch)):
                    self.pairs.append((pool[i % len(pool)], s3ch[i]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x128, x3 = self.pairs[idx]
        return torch.from_numpy(x128), torch.from_numpy(x3)


# ==================== OWN ====================

class SubjectDictBuilder:
    def __init__(self, dirs):
        self.dirs = dirs
        self.subjects = {}

    def load(self):
        for d in self.dirs:
            for fname in ["Ours_still_segments.npz", "Ours_sound_segments.npz"]:
                path = os.path.join(d, fname)
                if not os.path.exists(path):
                    continue
                data = np.load(path, allow_pickle=True)
                for sub_id in tqdm(data.files, desc=f"OWN {os.path.basename(d)} {fname}"):
                    item = data[sub_id].item()
                    segs = item["segments"]
                    label = item["label"]
                    if sub_id not in self.subjects:
                        self.subjects[sub_id] = {"segments": [], "label": label}
                    for i in range(segs.shape[0]):
                        self.subjects[sub_id]["segments"].append(segs[i].astype(np.float32))
        return self.subjects

    def get_subject_list(self):
        ids = list(self.subjects.keys())
        labels = [self.subjects[s]["label"] for s in ids]
        return ids, labels


class SegmentDataset(Dataset):
    def __init__(self, subjects, sub_ids):
        self.segments = []
        self.labels = []
        for sid in sub_ids:
            sub = subjects[sid]
            for seg in sub["segments"]:
                self.segments.append(seg)
                self.labels.append(sub["label"])
        self.segments = np.array(self.segments, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.segments[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ==================== Loader helpers ====================

def get_hbn_pretrain_loader(batch_size=64):
    ds = HBN128Dataset()
    print(f"HBN pretrain samples: {len(ds)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

def get_hbn_distill_loader(batch_size=128):
    ds = HBNDistillDataset()
    print(f"HBN distill pairs: {len(ds)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

def get_modma_teacher_dataloaders(batch_size=64, val_size=0.1, random_state=42):
    subjects = collect_modma_subjects()
    sub_ids = list(subjects.keys())
    sub_labels = [subjects[s]["label"] for s in sub_ids]
    num_classes = len(np.unique(sub_labels))
    train_ids, val_ids = train_test_split(
        sub_ids, test_size=val_size, random_state=random_state, stratify=sub_labels
    )
    train_data = {s: subjects[s]["segments"] for s in train_ids}
    train_lbls = {s: subjects[s]["label"] for s in train_ids}
    val_data   = {s: subjects[s]["segments"] for s in val_ids}
    val_lbls   = {s: subjects[s]["label"] for s in val_ids}

    train_ds = MODMAClassDataset(train_ids, train_data, train_lbls)
    val_ds   = MODMAClassDataset(val_ids, val_data, val_lbls)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"MODMA teacher: {len(train_ids)} train + {len(val_ids)} val subjects")
    print(f"  train labels: {np.bincount(list(train_lbls.values()))}")
    print(f"  val labels:   {np.bincount(list(val_lbls.values()))}")
    return train_loader, val_loader, num_classes

def get_modma_distill_loader(batch_size=128):
    ds = MODMADistillDataset()
    print(f"MODMA distill pairs: {len(ds)}")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

def get_own_dataloaders(batch_size=32, test_batch_size=32, val_size=0.2, random_state=42):
    builder = SubjectDictBuilder(TRAIN_DIRS)
    subjects = builder.load()
    sub_ids, sub_labels = builder.get_subject_list()
    print(f"OWN subjects: {len(sub_ids)}, labels: {np.bincount(sub_labels)}")
    train_ids, val_ids = train_test_split(
        sub_ids, test_size=val_size, random_state=random_state, stratify=sub_labels
    )
    train_set = SegmentDataset(subjects, train_ids)
    val_set   = SegmentDataset(subjects, val_ids)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=test_batch_size, shuffle=False)

    test_builder = SubjectDictBuilder(TEST_DIRS)
    test_subjects = test_builder.load()
    test_ids, test_labels = test_builder.get_subject_list()
    print(f"OWN test: {len(test_ids)} subjects, labels: {np.bincount(test_labels)}")
    test_set  = SegmentDataset(test_subjects, test_ids)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader