# DDL

**D**evin's **D**eep **L**earning Framework

#### 说明

+ ```datasets```包主要存储数据集、数据增强、任务配置等内容
+ ```models```包存储自行实现的模型，实现后请在main.py中的models_implemented加入名称以及工厂函数
+ ```utils```包用来存储损失函数、评价指标、训练器等内容
+ ```main.py```为整个程序的主入口，起到加载参数、数据集、模型、损失函数、模型训练等内容
+ ```parameter.yaml```为一个训练任务的参数配置，**请在运行前及时修改**

#### 使用方法（如果环境以满足要求可从3开始）

1. 确保安装好了```cuda、cudnn```，未安装的可以参照[torch-gpu版本安装教程](https://zhuanlan.zhihu.com/p/479848495)

2. 安装如下环境及依赖
   + ```python3.7.x (x >= 9)```
   + ```numpy==1.21.5```
   + ```torch==1.8及以上```
   + ```torchvision==0.9及以上```
   + ```tqdm==4.64.0```
   + ```matplotlib==3.5.3```
   + ```PyYAML==6.0```
   + ```tensorboard```
3. 安装好git后，使用如下命令克隆项目到本地（远端服务器）
   + 国内：```git clone https://gitee.com/devinmonster/ddl.git```
   + 国际：```git clone https://github.com/DevinMonster/ddl.git```
4. 修改```parameter.yaml```中的相关参数
5. 运行训练或测试```python main.py```