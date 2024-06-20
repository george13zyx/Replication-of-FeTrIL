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

![image](https://github.com/george13zyx/FeTrIL-/blob/main/78aca5e699006a96a28a6da03a9a27a.png)

