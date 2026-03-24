from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

    df.to_csv(filename)
    print()

# ================== 新增：通用路径解析与候选 ==================

def _find_split_from_path_parts(parts):
    lowers = [p.lower() for p in parts]
    for key in ["train", "valid", "val", "test"]:
        if key in lowers:
            return parts[lowers.index(key)]
    return "train"


def _parse_slide_id(slide_id):
    """解析 slide_id，返回 (split, body_part, patient_id, study_id)。兼容 \\ 与 /。
    例：MURA-v1.1/train/XR_ELBOW/patient00011/study1_negative/image1.png
    """
    s = str(slide_id).replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    if len(parts) > 0 and parts[0].startswith("MURA-v"):
        parts = parts[1:]
    split = _find_split_from_path_parts(parts)
    body_part = patient_id = study_id = None
    try:
        lowers = [p.lower() for p in parts]
        split_idx = lowers.index(split.lower())
        body_part = parts[split_idx + 1]
        patient_id = parts[split_idx + 2]
        study_id = parts[split_idx + 3]
    except Exception:
        if len(parts) >= 4:
            body_part = parts[-4]
            patient_id = parts[-3]
            study_id = parts[-2]
        else:
            body_part = parts[0] if len(parts) > 0 else "XR_UNKNOWN"
            patient_id = parts[1] if len(parts) > 1 else "patient00000"
            study_id = parts[2] if len(parts) > 2 else "study1"
    return split, body_part, patient_id, study_id


def _label_to_folder(label, num_classes=2):
    if num_classes == 2:
        return "abnormal" if int(label) == 1 else "normal"
    return str(int(label))


def _label_folder_candidates(label, num_classes=2):
    if num_classes == 2:
        if int(label) == 1:
            return ["abnormal", "positive", "1", "Abnormal", "Positive"]
        else:
            return ["normal", "negative", "0", "Normal", "Negative"]
    return [str(int(label)), _label_to_folder(label, num_classes)]


def _split_folder_candidates(split_name: str):
    s = str(split_name)
    cands = [s, s.lower(), s.capitalize()]
    sl = s.lower()
    if sl == "val":
        cands += ["valid", "Valid", "VAL"]
    elif sl == "valid":
        cands += ["val", "Val", "VALID"]
    # 去重
    seen = set(); out = []
    for x in cands:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def _candidate_filenames(body_part, patient_id, study_id, slide_id):
    s1 = f"{body_part}_{patient_id}_{study_id}.h5"
    s2 = f"{body_part}_{patient_id}_{study_id}.png.h5"
    s3 = str(slide_id).replace("/", "_").replace("\\", "_") + ".h5"
    s4 = f"{study_id}.h5"
    base_study = str(study_id).split("_")[0] if isinstance(study_id, str) else study_id
    s5 = f"{base_study}.h5"; s6 = f"{base_study}.png.h5"
    cands = [s1, s2, s3, s4, s5, s6]
    seen = set(); uniq = []
    for x in cands:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


def _try_resolve_path(root, folder, split, label_folder, candidates):
    attempts = []
    if not root:
        return None, attempts
    base = os.path.join(root, folder) if folder else root
    splits = split if isinstance(split, (list, tuple)) else [split]
    labels = label_folder if isinstance(label_folder, (list, tuple)) else [label_folder]
    for sp in splits:
        for lab in labels:
            for name in candidates:
                p = os.path.join(base, sp, lab, name) if lab else os.path.join(base, sp, name)
                attempts.append(p)
                if os.path.exists(p):
                    return p, attempts
        # 无标签兜底
        for name in candidates:
            p = os.path.join(base, sp, name)
            attempts.append(p)
            if os.path.exists(p):
                return p, attempts
    return None, attempts


def _fallback_legacy_paths(root, folder, body_part, patient_id, study_id, candidates):
    attempts = []
    if not root:
        return None, attempts
    base = os.path.join(root, folder) if folder else root
    for name in candidates:
        p = os.path.join(base, body_part, patient_id, name)
        attempts.append(p)
        if os.path.exists(p):
            return p, attempts
    for name in candidates:
        p = os.path.join(base, name)
        attempts.append(p)
        if os.path.exists(p):
            return p, attempts
    return None, attempts

# ================== 原有数据集定义 ==================

class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path = 'dataset_csv/ccrcc_clean.csv',
                 mode = 'clam',
                 shuffle = False,
                 seed = 7,
                 print_info = True,
                 label_dict = {},
                 filter_dict = {},
                 ignore=[],
                 patient_strat=False,
                 label_col = None,
                 patient_voting = 'max',
                 ):
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir_s = None
        self.data_dir_l = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id'])) 
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max() 
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])

        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
        settings = {
            'n_splits' : k,
            'val_num' : val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

        self.split_gen = generate_split(**settings)

    def set_splits(self,start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)

        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]

            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes)
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode, num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id=True, csv_path=None, data_root_dir=None, data_folder_s=None, data_folder_l=None, num_classes=2):
        # 保持原接口，新增可选参数以兼容上层调用
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode,  num_classes=self.num_classes)
            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, mode=self.mode,  num_classes=self.num_classes)
            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir_s=self.data_dir_s, data_dir_l=self.data_dir_l, num_classes=self.num_classes)
            else:
                test_split = None
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  
            train_split = self.get_split_from_df(all_splits, 'train')  
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test') 

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):

        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
                              columns= columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        # assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 data_dir_s,
                 data_dir_l,
                 mode,
                 **kwargs):

        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir_s = data_dir_s
        self.data_dir_l = data_dir_l
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        if type(self.data_dir_s) == dict and type(self.data_dir_l) == dict:
            source = self.slide_data['source'][idx]
            data_dir_s = self.data_dir_s['source']
            data_dir_l = self.data_dir_l['source']
        else:
            data_dir_s = self.data_dir_s
            data_dir_l = self.data_dir_l

        if not self.use_h5:
            if self.data_dir_s and self.data_dir_l:
                if(self.mode == 'transformer'):
                    # 解析 slide_id
                    split, body_part, patient_id, study_id = _parse_slide_id(slide_id)
                    split_candidates = _split_folder_candidates(split)
                    label_folders = _label_folder_candidates(label, self.num_classes)
                    candidates = _candidate_filenames(body_part, patient_id, study_id, slide_id)

                    # 兼容两种传参方式：
                    # 1) data_dir_* 为 "root" + "folder" 的组合（上层负责拼）
                    # 2) data_dir_* 直接是完整绝对路径到特征根
                    if os.path.isabs(str(data_dir_s)) and os.path.basename(str(data_dir_s)):
                        root_s, folder_s = data_dir_s, ""
                    else:
                        root_s, folder_s = data_dir_s, ""
                    if os.path.isabs(str(data_dir_l)) and os.path.basename(str(data_dir_l)):
                        root_l, folder_l = data_dir_l, ""
                    else:
                        root_l, folder_l = data_dir_l, ""

                    # 低分辨率
                    full_path_s, attempts_s = _try_resolve_path(root_s, folder_s, split_candidates, label_folders, candidates)
                    if full_path_s is None:
                        full_path_s2, attempts_s2 = _fallback_legacy_paths(root_s, folder_s, body_part, patient_id, study_id, candidates)
                        attempts_s += attempts_s2
                        full_path_s = full_path_s2
                    if full_path_s is None:
                        raise FileNotFoundError(
                            "未找到低分辨率特征文件.\n" +
                            f"slide_id: {slide_id}\n" +
                            f"split candidates: {split_candidates}\n" +
                            f"label folders tried: {label_folders}\n" +
                            f"search root: {root_s}, folder: {folder_s}\n" +
                            f"filenames: {candidates}\n" +
                            "attempts (sample):\n" + "\n".join(attempts_s[:20])
                        )

                    # 高分辨率
                    full_path_l, attempts_l = _try_resolve_path(root_l, folder_l, split_candidates, label_folders, candidates)
                    if full_path_l is None:
                        full_path_l2, attempts_l2 = _fallback_legacy_paths(root_l, folder_l, body_part, patient_id, study_id, candidates)
                        attempts_l += attempts_l2
                        full_path_l = full_path_l2
                    if full_path_l is None:
                        raise FileNotFoundError(
                            "未找到高分辨率特征文件.\n" +
                            f"slide_id: {slide_id}\n" +
                            f"split candidates: {split_candidates}\n" +
                            f"label folders tried: {label_folders}\n" +
                            f"search root: {root_l}, folder: {folder_l}\n" +
                            f"filenames: {candidates}\n" +
                            "attempts (sample):\n" + "\n".join(attempts_l[:20])
                        )

                    with h5py.File(full_path_s, 'r') as scale_s:
                        features_s = torch.from_numpy(np.array(scale_s['features']))
                        coords_s = torch.from_numpy(np.array(scale_s['coords']))
                    with h5py.File(full_path_l, 'r') as scale_l:
                        features_l = torch.from_numpy(np.array(scale_l['features']))
                        coords_l = torch.from_numpy(np.array(scale_l['coords']))
                    return features_s, coords_s, features_l, coords_l, label
            else:
                return slide_id, label
        else:
            full_path = os.path.join(data_dir_s, 'h5_files', '{}.h5'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, label, coords


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir_s=None, data_dir_l=None, mode='clam', num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir_s = data_dir_s
        self.data_dir_l = data_dir_l
        self.mode = mode
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)
