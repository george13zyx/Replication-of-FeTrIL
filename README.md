##！！这只是一份论文复现！！

### 简介：

​	FeTrIL是一种新型类增量学习方法。它结合了固定的特征提取器和伪特征生成器来改进增量性能。这种方法使用几何转换产生伪特征，使用固定提取简化训练过程，并使用迁移学习的方法提取特征。迁移学习比蒸馏简单，内存需求更小。FeTrIL旨在提高增量学习的性能，同时降低其在大规模数据集上的局限性。与现有技术如SSRE和其他使用类原型的方法相比，FeTrIL在维持模型的稳定性和可塑性方面取得了更好的平衡。

### 环境配置：

```bash
1. conda create -n fetril python=3.7

2. conda activate fetril

3. conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
//如果遇到下载速度很慢的问题
//选择使用以下命令
pip install torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116

4. pip install typing-extensions --upgrade

5. conda install pandas

6. pip install -U scikit-learn scipy matplotlib
```

### 代码运行流程：

格式：python -文件地址 -参数配置文件地址
eg：`python codes/scratch.py configs/cifar100.cf`
根据图中1-7的顺序执行代码，便可以得到最终结果

![image](https://github.com/george13zyx/FeTrIL-/blob/main/picture.png)

### 数据集的处理

我们在实验中采用了cifar100和tiny_imagenet这两个数据集，为了简便快捷的导入数据集，我们对数据集进行train和test的分类，并且以lst的文件格式导入到代码中。

由于我们使用以下代码读入数据集：

```python
    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, range_classes=None, random_seed=-1, old_load=False, open_images=False, num_labels=None):
        # images_list_file is a the root path of the list
        # transform is the transform to apply to the image
        # target_transform is the transform to apply to the target
        # return_path is a boolean to return the path of the image
        # range_classes is a range of the classes to take
        # random_seed is the seed to shuffle the classes
        # old_load is a boolean to load the list with the old way (i.e. without the root folder as first line)
        # open_images is a boolean to load the list with the open images format
        # num_labels is the number of labels to load (in case of more than 2 labels)
        self.return_path = return_path
        samples = []
        ####################################################################
        df = pd.read_csv(images_list_file, sep=' ', names=['paths','class']) # 注意此行
        ####################################################################
        if old_load:
            root_folder = ''
        else:
            root_folder = df['paths'].head(1)[0]
            df = df.tail(df.shape[0] -1)
        df.drop_duplicates()
        # cast the class to tuple of int
        if num_labels is None:
            df['class'] = df['class'].apply(lambda x: tuple(map(int, x.split(','))))
        else:
            df['class'] = df['class'].apply(lambda x: tuple(map(int, x.split(',')[:num_labels])))
        #print(df)
        df = df.sort_values('class')
        order_list = list(set(df['class'].values.tolist()))
        # sort the list by the first item of the tuple
        order_list.sort(key=lambda x: x[0])
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        print('Class order of',images_list_file,'before shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        deep_copy = order_list.copy()
        if random_seed != -1:
            np.random.seed(random_seed)
            random.shuffle(order_list)
            order_list_raw = np.random.permutation(len(order_list)).tolist()
            order_list = [deep_copy[i] for i in order_list_raw]
        print('Class order of',images_list_file,'after  shuffle with seed',random_seed,': [',*order_list[:5],'...',*order_list[-5:],']')
        print('*'*(len(images_list_file)+76+len(str(random_seed))))
        order_list_reverse = [deep_copy[order_list.index(i)] for i in list(set(df['class'].values.tolist()))]
        if range_classes:
            index_to_take = [order_list[i] for i in range_classes]
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1][0]]) for elt in list(map(tuple, df.loc[df['class'].isin(index_to_take)].values.tolist()))]
            samples.sort(key=lambda x:x[1])
            print('We pick the classes from', min(range_classes), 'to', max(range_classes), 'and we have', len(samples), 'samples')
        else:
            samples = [(os.path.join(root_folder, elt[0]),order_list_reverse[elt[1][0]]) for elt in list(map(tuple, df.values.tolist()))]
            samples.sort(key=lambda x:x[1])
            print('We have', len(samples), 'samples')
            print("first 3 samples",samples[:3])
        if not samples:
            raise(RuntimeError("No image found"))

        self.images_list_file = images_list_file
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = [s[0] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
```

所以我们的lst文件的输入格式一定为：路径+空格+标签

### 对cifar100数据集的处理：

下载好的CIFAR100数据集解压后，可以看到一共有四个文件，分别是：meta、train、test、file.txt~

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/1.png)

我们编写以下脚本进行cifar100数据集的处理：

```python
import os

def generate_lst(image_dir, output_file):
    with open(output_file, 'w') as file:
        index = 0
        for label_name in sorted(os.listdir(image_dir)):
            # Assuming folder names are like 'character_0', 'character_1', ...
            # Extract the label from the folder name
            label_idx = int(label_name.split('_')[-1])
            subdir = os.path.join(image_dir, label_name)
            if os.path.isdir(subdir):
                for img_name in sorted(os.listdir(subdir)):
                    img_path = os.path.join(subdir, img_name)
                    # Make sure we only write lines for actual image files
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        line = f"{img_path} {label_idx}\n"
                        file.write(line)
                        index += 1

# Set the directories for train and test images
train_dir = 'C:/Users/zhouyx/Desktop/cifar/train'  # Replace with the correct path to the training images
test_dir = 'C:/Users/zhouyx/Desktop/cifar/test'  # Replace with the correct path to the testing images

# Generate the .lst files for train and test images
generate_lst(train_dir, 'train.lst')
generate_lst(test_dir, 'test.lst')
```

处理后我们得到以下格式的lst文件：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/2.png)

我们编写以下config配置文件用于在scratch.py文件中读入数据集以及数据集的训练参数：

```
[config]
nb_classes        = 100
dataset           = cifar100
first_batch_size  = 50
il_states         = 10
random_seed       = -1
num_workers       = 12
regul             = 1.0
toler             = 0.0001
epochs_lucir      = 90
epochs_augmix_ft  = 290
list_root         = C:\Users\zhouyx\Desktop\FeTrIL-main\configs
model_root        = C:\Users\zhouyx\Desktop\FeTrIL-main\configs
feat_root         = C:\Users\zhouyx\Desktop\FeTrIL-main\configs
classifiers_root  = data/classifiers_cifar100
pred_root         = data/predictions_cifar100
mean_std          = C:\Users\zhouyx\Desktop\FeTrIL-main\configs\cifar100\data\mean_std
```

其中，mean_std需要在网上寻找该数据集的均值和方差，以文件形式存储在上述路径：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/3.png)

至此，cifar100数据集的处理结束

### 对tiny_imagenet数据集的处理：

下载好tiny_imagenet数据集并解压后，得到以下文件：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/4.png)

我们首先处理train的部分，我们打开train数据集，结构如下：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/5.png)

由于我们编写的读入数据集的代码要求label必须是int类型的数据，所以我们编写了以下代码进行类名的映射，映射范围为0-199：

```python
import os

# 定义要遍历的根目录和输出文件名
root_dir = 'c:\\Users\\zhouyx\\Downloads\\Compressed\\tiny-imagenet-200_2'
output_file = 'processed_data.txt'

# 初始化一个空的映射字典和标签计数器
label_mapping = {}
current_label = 0

# 打开输出文件
with open(output_file, 'w') as outfile:
    # 遍历根目录中的所有文件夹和文件
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # 过滤掉非JPEG文件
            if not file.endswith('.JPEG'):
                continue
            # 构建文件的绝对路径
            file_path = os.path.join(subdir, file)
            # 提取文件名中第二列字符串
            # 假设文件名格式为 val_XXXX.JPEG nXXXXX
            parts = file.split('.')
            second_column_string = parts[1].strip()  # 示例，实际情况请替换为正确的值

            # 如果这个字符串还没有映射，则分配一个新的整数标签
            if second_column_string not in label_mapping:
                label_mapping[second_column_string] = current_label
                current_label += 1

            # 获取这个字符串对应的整数标签
            label = label_mapping[second_column_string]

            # 写入处理后的数据到输出文件中
            outfile.write(f'{file_path} {label}\n')

print("处理完成，结果保存在 'processed_data.txt'")
print("标签映射如下:")
for k, v in label_mapping.items():
    print(f"{k}: {v}")
```

映射结果如下：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/6.png)

对于测试集，我们选用有标注的val而不是无标注的test文件夹，可以便于我们计算准确率：

```python
import os

# 读取数据文件
input_file = 'C:/Users/zhouyx/Downloads/Compressed/tiny-imagenet-200_2/tiny-imagenet-200/val/val_annotations.txt'
output_file = 'processed_data_1.txt'

# 获取当前工作目录的绝对路径
current_dir = os.getcwd()

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 将每行按空格分隔
        parts = line.strip().split()
        # 获取图片文件名
        image_file = parts[0]
        
        # 转换为绝对路径
        absolute_path = os.path.join(current_dir, image_file)
        # 获取类别
        category = parts[1][2:]
        # 写入处理后的数据
        outfile.write(f'{absolute_path} {category}\n')

print("处理完成，结果保存在 'processed_data.txt'")
```

通过处理好的标签，使用train数据集的映射进行处理：

```python
# 定义输入文件名和输出文件名
input_file = 'C:/Users/zhouyx/Downloads/Compressed/tiny-imagenet-200_2/test_remake.txt'
output_file = 'output_test.txt'

# 定义标签映射字典
label_mapping = {
    1443537: 0, 1629819: 1, 1641577: 2, 1644900: 3, 1698640: 4, 1742172: 5, 1768244: 6, 1770393: 7,
    1774384: 8, 1774750: 9, 1784675: 10, 1855672: 11, 1882714: 12, 1910747: 13, 1917289: 14, 1944390: 15,
    1945685: 16, 1950731: 17, 1983481: 18, 1984695: 19, 2002724: 20, 2056570: 21, 2058221: 22, 2074367: 23,
    2085620: 24, 2094433: 25, 2099601: 26, 2099712: 27, 2106662: 28, 2113799: 29, 2123045: 30, 2123394: 31,
    2124075: 32, 2125311: 33, 2129165: 34, 2132136: 35, 2165456: 36, 2190166: 37, 2206856: 38, 2226429: 39,
    2231487: 40, 2233338: 41, 2236044: 42, 2268443: 43, 2279972: 44, 2281406: 45, 2321529: 46, 2364673: 47,
    2395406: 48, 2403003: 49, 2410509: 50, 2415577: 51, 2423022: 52, 2437312: 53, 2480495: 54, 2481823: 55,
    2486410: 56, 2504458: 57, 2509815: 58, 2666196: 59, 2669723: 60, 2699494: 61, 2730930: 62, 2769748: 63,
    2788148: 64, 2791270: 65, 2793495: 66, 2795169: 67, 2802426: 68, 2808440: 69, 2814533: 70, 2814860: 71,
    2815834: 72, 2823428: 73, 2837789: 74, 2841315: 75, 2843684: 76, 2883205: 77, 2892201: 78, 2906734: 79,
    2909870: 80, 2917067: 81, 2927161: 82, 2948072: 83, 2950826: 84, 2963159: 85, 2977058: 86, 2988304: 87,
    2999410: 88, 3014705: 89, 3026506: 90, 3042490: 91, 3085013: 92, 3089624: 93, 3100240: 94, 3126707: 95,
    3160309: 96, 3179701: 97, 3201208: 98, 3250847: 99, 3255030: 100, 3355925: 101, 3388043: 102, 3393912: 103,
    3400231: 104, 3404251: 105, 3424325: 106, 3444034: 107, 3447447: 108, 3544143: 109, 3584254: 110, 3599486: 111,
    3617480: 112, 3637318: 113, 3649909: 114, 3662601: 115, 3670208: 116, 3706229: 117, 3733131: 118, 3763968: 119,
    3770439: 120, 3796401: 121, 3804744: 122, 3814639: 123, 3837869: 124, 3838899: 125, 3854065: 126, 3891332: 127,
    3902125: 128, 3930313: 129, 3937543: 130, 3970156: 131, 3976657: 132, 3977966: 133, 3980874: 134, 3983396: 135,
    3992509: 136, 4008634: 137, 4023962: 138, 4067472: 139, 4070727: 140, 4074963: 141, 4099969: 142, 4118538: 143,
    4133789: 144, 4146614: 145, 4149813: 146, 4179913: 147, 4251144: 148, 4254777: 149, 4259630: 150, 4265275: 151,
    4275548: 152, 4285008: 153, 4311004: 154, 4328186: 155, 4356056: 156, 4366367: 157, 4371430: 158, 4376876: 159,
    4398044: 160, 4399382: 161, 4417672: 162, 4456115: 163, 4465501: 164, 4486054: 165, 4487081: 166, 4501370: 167,
    4507155: 168, 4532106: 169, 4532670: 170, 4540053: 171, 4560804: 172, 4562935: 173, 4596742: 174, 4597913: 175,
    6596364: 176, 7579787: 177, 7583066: 178, 7614500: 179, 7615774: 180, 7695742: 181, 7711569: 182, 7715103: 183, 7720875: 184, 7734744: 185, 7747607: 186, 7749582: 187, 7753592: 188, 7768694: 189,
7871810: 190, 7873807: 191, 7875152: 192, 7920052: 193, 9193705: 194, 9246464: 195, 9256479: 196, 9332890: 197,
9428293: 198, 12267677: 199}

# 读取输入文件，转换标签，并写入输出文件
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        parts = line.split()
        if len(parts) >= 2:
            img_path, label_str = parts[0], parts[1]
            label = label_mapping.get(int(label_str), -1)
            if label != -1:
                f_out.write(f'{img_path} {label}\n')
            else:
                print(f'Invalid label: {label_str}')
        else:
            print(f'Invalid line: {line}')
```

最终处理结果如下：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/7.png)

我们编写以下config配置文件用于在scratch.py文件中读入数据集以及数据集的训练参数：

```
[config]
nb_classes        = 200
dataset           = tiny_imagenet
first_batch_size  = 50
il_states         = 10
random_seed       = -1
num_workers       = 12
regul             = 1.0
toler             = 0.0001
epochs_lucir      = 30
epochs_augmix_ft  = 100
list_root         = C:\Users\zhouyx\Desktop\FeTrIL-main\configs
model_root        = C:\Users\zhouyx\Desktop\FeTrIL-main\configs
feat_root         = C:\Users\zhouyx\Desktop\FeTrIL-main\configs
classifiers_root  = data/classifiers_tiny_imagenet
pred_root         = data/predictions_tiny_imagenet
mean_std          = C:\Users\zhouyx\Desktop\FeTrIL-main\configs\tiny_imagenet\data\mean_std
```

其中，mean_std需要在网上寻找该数据集的均值和方差，以文件形式存储在上述路径：

![image](https://github.com/george13zyx/FeTrIL/blob/main/images/8.png)

至此，tiny_imagenet数据集的处理结束
