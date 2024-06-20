from configparser import ConfigParser
from multiprocessing import Pool
import os
import sys

def decompose_class(args):
    root_path, n = args
    file_path = os.path.join(root_path, str(n))
    if os.path.exists(file_path):
        try:
            # 创建分解后的文件目录（如果不存在）
            os.makedirs(file_path + '_decomposed', exist_ok=True)
            compteur = 0
            # 读取原文件并逐行分解
            with open(file_path, 'r') as f:
                for line in f:
                    with open(os.path.join(file_path + '_decomposed', str(compteur)), 'w') as f2:
                        f2.write(line)
                    compteur += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

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
    first_batch_size = int(cp["first_batch_size"])
    il_states = int(cp["il_states"])
    feat_root = cp["feat_root"]
    incr_batch_size = (nb_classes - first_batch_size) // il_states

    for state_id in range(il_states + 1):
        print("Preparing state", state_id, "of", il_states)
        # 构造当前批次的路径
        root_path = os.path.join(feat_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size), "t" + str(il_states), "train", "batch" + str(state_id))

        # 计算当前的类别数
        current_nb_classes = first_batch_size + state_id * incr_batch_size
        # 使用多进程处理当前类别
        with Pool() as p:
            p.map(decompose_class, [(root_path, n) for n in range(current_nb_classes)])