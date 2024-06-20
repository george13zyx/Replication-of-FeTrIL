from configparser import ConfigParser
from multiprocessing import Pool
import os
import shutil
import sys

# 定义处理类的函数，用于删除生成的 '_decomposed' 文件夹
def decompose_class(args):
    root_path, n = args
    file_path = os.path.join(root_path, str(n))
    try:
        # 尝试删除 '_decomposed' 文件夹
        shutil.rmtree(file_path + '_decomposed')
    except Exception as e:
        # 如果删除失败，打印错误信息
        print(f"Error removing {file_path + '_decomposed'}: {e}")

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
    nb_classes = int(cp['nb_classes'])
    dataset = cp['dataset']
    random_seed = int(cp["random_seed"])
    first_batch_size = int(cp["first_batch_size"])
    il_states = int(cp["il_states"])
    feat_root = cp["feat_root"]
    incr_batch_size = (nb_classes - first_batch_size) // il_states

    # 遍历每个状态，清理 '_decomposed' 文件夹
    for state_id in range(il_states + 1):
        print("Cleaning state", state_id, "of", il_states)
        root_path = os.path.join(feat_root, "fetril", dataset, "seed" + str(random_seed), "b" + str(first_batch_size), "t" + str(il_states), "train", "batch" + str(state_id))

        nb_classes = first_batch_size + (state_id) * incr_batch_size

        # 使用多进程并行处理每个类的删除操作
        with Pool() as p:
            p.map(decompose_class, [(root_path, n) for n in range(nb_classes)])
