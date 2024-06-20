from configparser import ConfigParser
import os
import sys
import csv
import numpy as np
from multiprocessing import Pool, freeze_support


# 定义扁平化列表的函数
def flatten(t):
    return [item for sublist in t for item in sublist]


# 定义计算评分的函数
def compute_score(args):
    nb_batch, root_path_pred, first_batch_size, batch_size = args
    path_pred = os.path.join(root_path_pred, "batch" + str(nb_batch))
    y_pred = []
    score_top5 = []
    y_true = []

    # 读取预测文件并计算top5分数
    for c in range(first_batch_size + batch_size * nb_batch):
        with open(os.path.join(path_pred, str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])

    y_pred = np.asarray(flatten(y_pred))
    y_pred_top5 = flatten(score_top5)
    y_true = np.asarray(flatten(y_true))
    return (nb_batch, [np.mean(y_pred == y_true), np.mean(y_pred_top5)])


# 定义详细评分的函数
def detailled_score(args):
    nb_batch, root_path_pred, first_batch_size, batch_size = args
    path_pred = os.path.join(root_path_pred, "batch" + str(nb_batch))
    res = []

    # 读取预测文件并计算每个批次的分数
    for c in range(first_batch_size + batch_size * nb_batch):
        y_pred = []
        score_top5 = []
        y_true = []

        with open(os.path.join(path_pred, str(c)), newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            to_append_top5 = [[int(elt[i].split(":")[0]) for i in range(5)] for elt in list(reader)]
            to_append = [elt[0] for elt in to_append_top5]
            y_pred.append(to_append)
            score_top5.append([c in to_append_top5[i] for i in range(len(to_append))])
            y_true.append([c for _ in to_append])

        y_pred = np.asarray(flatten(y_pred))
        y_pred_top5 = flatten(score_top5)
        y_true = np.asarray(flatten(y_true))
        prdic = np.mean(y_pred == y_true)
        res.append(prdic)

    return ([np.mean(res[:first_batch_size])] + [np.mean(res[i:i + batch_size]) for i in
                                                 range(first_batch_size, len(res), batch_size)])


# 主函数部分
if __name__ == '__main__':
    freeze_support()

    # 检查命令行参数，如果参数不正确则退出
    if len(sys.argv) != 2:
        print('Arguments: config')
        sys.exit(-1)

    # 读取配置文件
    cp = ConfigParser()
    with open(sys.argv[1]) as fh:
        cp.read_file(fh)
    cp = cp['config']

    # 从配置文件中读取相关参数
    nb_classes = int(cp['nb_classes'])
    dataset = cp['dataset']
    first_batch_size = int(cp["first_batch_size"])
    il_states = int(cp["il_states"])
    random_seed = int(cp["random_seed"])
    feat_root = cp["feat_root"]
    pred_root = cp["pred_root"]
    classifiers_root = cp["classifiers_root"]
    t = il_states
    root_path_pred = os.path.join(pred_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size),
                                  "t" + str(il_states))

    # 计算每个批次的大小
    batch_size = (nb_classes - first_batch_size) // il_states

    batches = range(t + 1)
    resultats = {}

    # 构建参数列表
    args = [(nb_batch, root_path_pred, first_batch_size, batch_size) for nb_batch in batches]

    # 使用多进程并行处理每个批次的评分计算
    with Pool() as p:
        resultats = dict(p.map(compute_score, args))

    top1 = []
    top5 = []
    # 收集每个批次的top1和top5的平均值
    for batch_number in batches:
        top1.append(resultats[batch_number][0])
        top5.append(resultats[batch_number][1])

    # 输出top1和top5的平均值
    print("top1:", [round(100 * elt, 2) for elt in top1])
    print("top5:", [round(100 * elt, 2) for elt in top5])
    print(f'top1 = {sum(top1) / len(top1):.3f}, top5 = {sum(top5) / len(top5):.3f}')


