from multiprocessing import Pool
import torch
import sys
import os
from sklearn.svm import LinearSVC
from configparser import ConfigParser
import numpy as np
import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
import itertools

# 添加自定义路径到系统路径
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'utilsCIL'))

# 定义 L3 特征数据集类
class L3FeaturesDataset(Dataset):
    def __init__(self, path, range_classes=None):
        self.y = []
        self.X = []
        doss = os.listdir(path)
        if range_classes:
            doss = [f'{d}_decomposed' for d in range_classes]
        doss = np.sort(doss)
        for root_path in doss:
            if '_' in root_path:
                for file in os.listdir(os.path.join(path, root_path)):
                    self.y.append(int(root_path.split('_')[0]))
                    self.X.append(os.path.join(path, root_path, file))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).reshape(256,14,14).astype(np.float32)
        return image, label

# 定义 L3 特征映射数据集类
class L3FeaturesDatasetFeatureMaps(Dataset):
    def __init__(self, path, range_classes=None):
        self.y = []
        self.X = []
        doss = os.listdir(path)
        if range_classes:
            doss = [f'{d}_decomposed' for d in range_classes]
        doss = np.sort(doss)
        for root_path in doss:
            if '_' in root_path:
                for file in os.listdir(os.path.join(path, root_path)):
                    self.y.append(int(root_path.split('_')[0]))
                    self.X.append(os.path.join(path, root_path, file))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).reshape(14,256*14).astype(np.float32)
        return image, label

# 定义 L4 特征数据集类
class L4FeaturesDataset(Dataset):
    def __init__(self, path, range_classes=None):
        self.y = []
        self.X = []
        doss = os.listdir(path)
        if range_classes:
            doss = [f'{d}_decomposed' for d in range_classes]
        doss = np.sort(doss)
        for root_path in doss:
            if '_' in root_path:
                for file in os.listdir(os.path.join(path, root_path)):
                    self.y.append(int(root_path.split('_')[0]))
                    self.X.append(os.path.join(path, root_path, file))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        label = self.y[index]
        with open(self.X[index], "r") as f:
            image = np.array([float(x) for x in f.read().split()]).astype(np.float32)
        return image, label

# 定义规范化训练特征的函数
def normalize_train_features(il_dir, state_id, state_size, first_batch_size, nb_classes):
    feats_libsvm = []
    min_pos = 0
    max_pos = first_batch_size + state_id * state_size
    current_range_classes = range(min_pos, max_pos)
    class_list = list(range(nb_classes))
    train_dataset = L4FeaturesDataset(il_dir, range_classes=current_range_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=0, pin_memory=True)
    matrix_mix = list(iter(train_loader))
    matrix = matrix_mix[0][0]
    classes = list(matrix_mix[0][1])
    for i in range(1, len(matrix_mix)):
        matrix = torch.vstack((matrix, torch.nn.functional.normalize(matrix_mix[i][0], p=2, dim=1)))
        classes.extend(matrix_mix[i][1])
    feats_libsvm = matrix.numpy()
    return np.array(classes), feats_libsvm

# 定义计算阈值的函数
def calc_thrd(args):
    crt_id, y_true, X, classifiers_dir, regul, toler = args
    crt_id = str(crt_id)
    crt_id_svm_path = os.path.join(classifiers_dir, crt_id + ".model")
    if not os.path.exists(crt_id_svm_path):
        y = np.empty(y_true.shape, dtype=str)
        y[y_true == crt_id] = '+1'
        y[y_true != crt_id] = '-1'
        clf = LinearSVC(penalty='l2', dual=False, tol=float(toler), C=float(regul), multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0)
        clf.fit(X, y)
        svm_weights = clf.coef_
        svm_bias = clf.intercept_
        out_weights = " ".join(map(str, svm_weights.flatten()))
        out_bias = str(svm_bias.item(0))
        with open(crt_id_svm_path, "w") as f_svm:
            f_svm.write(out_weights + "\n")
            f_svm.write(out_bias + "\n")

# 检查命令行参数是否正确
if len(sys.argv) != 2:
    print('Arguments: config')
    sys.exit(-1)

# 读取配置文件
cp = ConfigParser()
with open(sys.argv[1]) as fh:
    cp.read_file(fh)
cp = cp['config']
regul = float(cp["regul"])
toler = float(cp["toler"])
random_seed = int(cp["random_seed"])
nb_classes = int(cp['nb_classes'])
dataset = cp['dataset']
first_batch_size = int(cp["first_batch_size"])
il_states = int(cp["il_states"])
feat_root = cp["feat_root"]
classifiers_root = cp["classifiers_root"]

incr_batch_size = (nb_classes - first_batch_size) // il_states

# 主函数部分
if __name__ == '__main__':
    for state_id in range(il_states + 1):
        print("Training state", state_id, "of", il_states)
        il_dir = os.path.join(feat_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size), "t" + str(il_states), "train", "batch" + str(state_id))
        classifiers_dir = os.path.join(classifiers_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size), "t" + str(il_states), "batch" + str(state_id))
        min_pos = 0
        max_pos = first_batch_size + state_id * incr_batch_size
        state_size = first_batch_size
        if state_id > 0:
            state_size = incr_batch_size
        to_check = any([not os.path.exists(os.path.join(classifiers_dir, str(crt_id) + ".model")) for crt_id in range(min_pos, max_pos)])
        if not to_check:
            print("Classifiers already created for range", [os.path.join(classifiers_dir, str(crt_id) + ".model") for crt_id in [min_pos, max_pos - 1]])
        else:
            y_true, norm_feats = normalize_train_features(il_dir, state_id, state_size, first_batch_size, nb_classes)
            df = pd.DataFrame(norm_feats, columns=['feat' + str(i + 1) for i in range(norm_feats.shape[1])])
            y_true = np.array(y_true, dtype=str)
            X = df.to_numpy(dtype=float)
            os.makedirs(classifiers_dir, exist_ok=True)

            with Pool() as p:
                p.map(calc_thrd, [(crt_id, y_true, X, classifiers_dir, regul, toler) for crt_id in range(min_pos, max_pos)])