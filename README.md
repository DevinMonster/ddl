# DDL
Devin's Deep Learning Modules

#### 说明
+ ```datasets```包主要存储数据集相关与数据增强相关类，以及与分割相关的任务配置
+ ```models```包主要用来存储模型相关内容
+ ```utils```包用来存储损失函数、日志等内容
+ ```tests```包主要用来测试模块的正确性
+ ```main.py```为整个程序的主入口，起到加载参数、数据集、模型、损失函数、模型训练等内容
+ ```parameter.yaml```为一个训练任务的参数配置，请在**运行前及时修改**
#### 使用方法
1. 克隆项目到本地
  + ```git clone https://github.com/DevinMonster/ddl.git```
2. 修改```parameter.yaml```中的相关参数
3. 运行训练或测试
   + ```python main.py```