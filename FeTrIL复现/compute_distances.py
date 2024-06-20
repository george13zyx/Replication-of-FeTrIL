from configparser import ConfigParser
import numpy as np
import os
import csv
import sys
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool


def compute_batch(args):
    curr_state, train_data_path_lucir, train_data_path, first_batch_size, batch_size = args
    # 创建当前批次的目录（如果不存在）
    os.makedirs(os.path.join(train_data_path_lucir, "batch" + str(curr_state)), exist_ok=True)
    X_val, y_val = None, None

    for data_path in set([train_data_path]):
        chaton = []
        poney = []
        # 遍历当前批次的数据
        for i in range(first_batch_size + curr_state * batch_size):
            path_to_data = data_path + str(i)
            to_np = []
            # 读取数据文件并转换为numpy数组
            f_data = open(path_to_data)
            for dline in f_data:
                dline = dline.rstrip()
                to_np.append(np.fromstring(dline, sep=' ', dtype=float))
            f_data.close()

            with open(path_to_data, newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                data = np.array(list(reader), dtype=float)
            chaton.append(data)
            poney += [i for _ in range(len(data))]

        if data_path == train_data_path:
            X_train = np.concatenate(chaton)
            y_train = np.array(poney)

    X = X_train
    y = y_train

    # 计算每类数据的均值
    means = [np.mean(X[y == i], axis=0) for i in set(y)]
    if curr_state == 0:
        means_curr_state = means[:first_batch_size]
    else:
        means_curr_state = means[
                           first_batch_size + (curr_state - 1) * batch_size: first_batch_size + curr_state * batch_size]

    # 计算均值之间的距离
    distances = pairwise_distances(means, means_curr_state)
    if curr_state == 0:
        c_distance_mini = np.argmin(distances, axis=1)
    else:
        c_distance_mini = np.argmin(distances, axis=1) + first_batch_size + (curr_state - 1) * batch_size

    Xp = []
    Yp = []
    for c in set(y):
        # 如果当前类别的文件不存在，计算并保存数据
        if not os.path.exists(os.path.join(train_data_path_lucir, "batch" + str(curr_state), str(c))):
            X_tmp = X[y == c_distance_mini[c]] - np.expand_dims(means[c_distance_mini[c]], 0) + np.expand_dims(means[c],
                                                                                                               0)
            print("saving batch", curr_state, "- class", c)
            np.savetxt(os.path.join(train_data_path_lucir, "batch" + str(curr_state), str(c)), X_tmp, fmt='%1.8f')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Arguments: config')
        sys.exit(-1)

    cp = ConfigParser()
    # 读取配置文件
    with open(sys.argv[1]) as fh:
        cp.read_file(fh)
    cp = cp['config']

    # 读取配置参数
    nb_classes = int(cp['nb_classes'])
    dataset = cp['dataset']
    random_seed = int(cp["random_seed"])
    feat_root = cp['feat_root']
    il_states = int(cp['il_states'])
    first_batch_size = int(cp['first_batch_size'])
    t = il_states

    # 设置训练数据路径和目标路径
    train_data_path = os.path.join(feat_root, dataset, "seed" + str(random_seed), "b" + str(first_batch_size), "train/")
    train_data_path_lucir = os.path.join(feat_root, "fetril", dataset, "seed" + str(random_seed),
                                         "b" + str(first_batch_size), "t" + str(il_states), "train/")

    # 计算每批次的大小
    batch_size = (nb_classes - first_batch_size) // t

    to_compute = []
    # 确定需要计算的批次
    for i in range(t + 1):
        to_check = any([not os.path.exists(os.path.join(train_data_path_lucir, "batch" + str(i), str(c))) for c in
                        range(first_batch_size + (i) * batch_size)])
        if to_check:
            to_compute.append(i)
    print("ToC:", to_compute)

    # 构造需要计算的参数
    args = [(i, train_data_path_lucir, train_data_path, first_batch_size, batch_size) for i in to_compute]

    # 使用多进程计算批次
    with Pool() as p:
        p.map(compute_batch, args)
