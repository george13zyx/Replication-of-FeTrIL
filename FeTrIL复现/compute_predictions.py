# 导入必要的库
from configparser import ConfigParser
import sys
import os
import numpy as np
from sklearn.preprocessing import Normalizer
from multiprocessing import Pool


# 定义计算特征的函数
def compute_feature(args):
    i, first_batch_size, nb_classes, T, test_feats_path, pred_path, model_dir = args
    # 计算对应的批次
    corresponding_batch = (i - first_batch_size) // ((nb_classes - first_batch_size) // T) + 1
    if i < first_batch_size:
        corresponding_batch = 0
    test_feats = os.path.join(test_feats_path, str(i))

    # 为每个批次创建预测文件
    for batchs in range(corresponding_batch, T + 1):
        os.makedirs(os.path.join(pred_path, "batch" + str(batchs)), exist_ok=True)
        pred_file = os.path.join(pred_path, "batch" + str(batchs), str(i))

        # 如果预测文件不存在，则进行计算
        if not os.path.exists(pred_file):
            with open(pred_file, "w") as f_pred:
                syns = []
                f_list_syn = list(range(((nb_classes - first_batch_size) // T) * (batchs) + first_batch_size))
                for syn in f_list_syn:
                    syn = str(syn)
                    syns.append(syn)
                weights_list = []
                biases_list = []

                # 读取每个模型的权重和偏置
                for syn in range(len(syns)):
                    line_cnt = 0
                    target_model = os.path.join(model_dir, "batch" + str(batchs), str(syn) + ".model")
                    f_model = open(target_model)
                    for line in f_model:
                        line = line.rstrip()
                        if line_cnt == 0:
                            parts = line.split(" ")
                            parts_float = [float(pp) for pp in parts]
                            weights_list.append(parts_float)
                        elif line_cnt == 1:
                            biases_list.append(float(line))
                        line_cnt += 1
                    f_model.close()

                # 读取测试特征并进行归一化处理
                f_test_feat = open(test_feats, 'r')
                for vline in f_test_feat.readlines():
                    vparts = vline.split(" ")
                    crt_feat = [[float(vp) for vp in vparts]]
                    crt_feat = Normalizer().fit_transform(crt_feat)[0]

                    # 计算每个类别的预测分数
                    pred_dict = []
                    for cls_cnt in range(len(weights_list)):
                        cls_score = np.dot(crt_feat, weights_list[cls_cnt]) + biases_list[cls_cnt]
                        pred_dict.append(-cls_score)

                    # 按分数排序并写入预测文件
                    pred_line = ""
                    predictions_idx = sorted(range(len(pred_dict)), key=lambda k: -pred_dict[k])
                    for idx in predictions_idx:
                        pred_line += " " + str(idx) + ":" + str(pred_dict[idx])
                    pred_line = pred_line.lstrip()
                    f_pred.write(pred_line + "\n")
                f_test_feat.close()
        else:
            print("exists predictions file:", pred_file)


# 主函数部分
if __name__ == '__main__':
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

    # 构建路径
    test_feats_path = os.path.join(feat_root, dataset, "seed" + str(random_seed), "b" + str(first_batch_size), "test/")
    svms_dir = os.path.join(classifiers_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size),
                            "t" + str(il_states))
    pred_path = os.path.join(pred_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size),
                             "t" + str(il_states))
    model_dir = svms_dir
    os.makedirs(pred_path, exist_ok=True)
    T = il_states

    # 构建参数列表
    args_list = [(i, first_batch_size, nb_classes, T, test_feats_path, pred_path, model_dir) for i in range(nb_classes)]

    # 使用多进程并行处理每个特征计算
    with Pool() as p:
        p.map(compute_feature, args_list)
