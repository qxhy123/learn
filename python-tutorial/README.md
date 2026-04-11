# 从零到高阶的Python教程（深度学习应用版）

## 项目简介

本教程旨在为学习者提供一套系统、完整的Python学习资源，从最基础的语法出发，循序渐进地覆盖面向对象编程、函数式编程、科学计算、数据可视化等核心主题，最终深入到深度学习框架PyTorch的应用、模型部署，以及 `asyncio` 驱动的高并发系统实践。

**本教程的独特之处**：每章都包含「深度学习应用」部分，展示该章Python知识在人工智能和机器学习领域的实际应用，配有可运行的PyTorch代码示例。

---

## 目标受众

- 零基础编程初学者，希望系统学习Python
- 有其他编程语言经验，希望快速掌握Python的开发者
- 数据科学、机器学习领域的学习者和从业者
- 希望深入理解PyTorch底层原理的深度学习工程师
- 大学计算机相关专业的在校学生

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：Python基础

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第1章 | [Python环境与基本语法](./part1-basics/01-environment-syntax.md) | 安装配置、变量、数据类型、运算符 | 张量数据类型 |
| 第2章 | [控制流与函数](./part1-basics/02-control-flow-functions.md) | 条件语句、循环、函数定义与调用 | 训练循环控制 |
| 第3章 | [数据结构](./part1-basics/03-data-structures.md) | 列表、元组、字典、集合 | 批量数据组织 |
| 第4章 | [字符串与文件操作](./part1-basics/04-strings-files.md) | 字符串方法、文件读写、路径处理 | 数据集加载 |

### 第二部分：Python进阶

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第5章 | [面向对象编程](./part2-intermediate/05-oop.md) | 类与对象、继承、多态、魔术方法 | nn.Module设计 |
| 第6章 | [模块与包](./part2-intermediate/06-modules-packages.md) | 模块导入、包结构、虚拟环境 | PyTorch项目组织 |
| 第7章 | [异常处理与调试](./part2-intermediate/07-exceptions-debugging.md) | try/except、日志、调试技巧 | 训练异常处理 |
| 第8章 | [迭代器与生成器](./part2-intermediate/08-iterators-generators.md) | 迭代协议、生成器、yield | 数据流水线 |

### 第三部分：函数式编程与高级特性

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第9章 | [函数式编程](./part3-functional/09-functional-programming.md) | lambda、map、filter、reduce | 数据变换管道 |
| 第10章 | [装饰器与闭包](./part3-functional/10-decorators-closures.md) | 装饰器原理、闭包、functools | 训练监控装饰器 |
| 第11章 | [上下文管理器与描述符](./part3-functional/11-context-descriptors.md) | with语句、描述符协议 | 自动混合精度 |

### 第四部分：科学计算基础

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第12章 | [NumPy基础](./part4-scientific/12-numpy-basics.md) | 数组创建、索引、运算、广播 | 张量操作基础 |
| 第13章 | [NumPy高级应用](./part4-scientific/13-numpy-advanced.md) | 线性代数、随机数、高级索引 | 权重初始化 |
| 第14章 | [Pandas数据处理](./part4-scientific/14-pandas.md) | DataFrame、数据清洗、分组聚合 | 数据预处理 |

### 第五部分：数据可视化

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第15章 | [Matplotlib基础](./part5-visualization/15-matplotlib.md) | 图表绑制、样式定制、子图布局 | 训练曲线可视化 |
| 第16章 | [高级可视化](./part5-visualization/16-advanced-visualization.md) | Seaborn、交互式图表、动画 | 特征可视化 |

### 第六部分：Python与深度学习

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第17章 | [PyTorch张量基础](./part6-pytorch/17-pytorch-tensors.md) | 张量创建、运算、自动微分 | 计算图构建 |
| 第18章 | [神经网络构建](./part6-pytorch/18-neural-networks.md) | nn.Module、常用层、参数管理 | 模型架构设计 |
| 第19章 | [训练循环与优化](./part6-pytorch/19-training-optimization.md) | 损失函数、优化器、训练流程 | 完整训练管道 |
| 第20章 | [数据加载与预处理](./part6-pytorch/20-data-loading.md) | Dataset、DataLoader、数据增强 | 高效数据管道 |

### 第七部分：深度学习实战

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第21章 | [图像分类实战](./part7-deep-learning/21-image-classification.md) | CNN架构、图像处理、迁移学习 | CIFAR-10分类 |
| 第22章 | [自然语言处理基础](./part7-deep-learning/22-nlp-basics.md) | 文本处理、词嵌入、文本分类 | 情感分析 |
| 第23章 | [序列模型与注意力机制](./part7-deep-learning/23-sequence-attention.md) | RNN、LSTM、Transformer | 机器翻译 |

### 第八部分：高级主题

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第24章 | [模型部署与优化](./part8-advanced/24-deployment.md) | ONNX导出、量化、模型服务 | 生产环境部署 |

### 第九部分：异步与高并发系统

| 章节 | 标题 | 主要内容 | 深度学习应用 |
|------|------|----------|--------------|
| 第25章 | [asyncio基础与结构化并发](./part9-asyncio/25-asyncio-foundations.md) | 协程、Task、TaskGroup、取消、超时、同步原语 | 异步特征抓取与批量推理控制面 |
| 第26章 | [asyncio网络编程与异步服务实战](./part9-asyncio/26-asyncio-networking-and-services.md) | `start_server`、背压、限流、优雅关闭、异步服务 | 异步模型网关与日志摄取服务 |
| 第27章 | [asyncio极端复杂场景实战](./part9-asyncio/27-asyncio-extreme-scenarios.md) | deadline预算、微批处理、多租户限流、取消风暴、复杂编排 | 高并发推理网关与RAG编排器 |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [Python速查表](./appendix/python-cheatsheet.md) | 常用语法、内置函数速查 |
| 附录B | [PyTorch常用API](./appendix/pytorch-api.md) | 张量操作、神经网络API速查 |
| 附录C | [练习答案汇总](./appendix/answers.md) | 各章练习题答案索引与新 async 章节要点 |

---

## 学习路径建议

### 路径一：零基础入门（约 6-8 周）

适合完全没有编程经验的初学者：

1. 系统学习第1-4章（Python基础）
2. 学习第5-8章（Python进阶）
3. 学习第12章（NumPy基础）
4. 学习第17章（PyTorch张量基础）

### 路径二：快速进阶（约 4-6 周）

适合有其他编程语言经验的学习者：

1. 快速浏览第1-4章，掌握Python特有语法
2. 重点学习第5章（面向对象）和第8-11章（高级特性）
3. 学习第12-14章（科学计算）
4. 深入学习第17-20章（PyTorch基础）

### 路径三：深度学习导向（约 3-4 周）

适合已有Python基础、希望学习深度学习的工程师：

1. 复习第5章（面向对象）和第8章（生成器）
2. 学习第12-13章（NumPy）
3. 系统学习第17-20章（PyTorch核心）
4. 实战第21-27章（深度学习项目、部署与异步系统编排）

### 路径四：后端 / Infra 导向（约 2-3 周）

适合希望把 Python 用到高并发服务、模型网关、异步编排的工程师：

1. 复习第2章（函数）、第7章（异常）、第8章（生成器）
2. 学习第10-11章（装饰器、上下文管理）
3. 学习第24章（部署）
4. 重点学习第25-27章（`asyncio` 与复杂系统场景）

---

## 前置要求

学习本教程的前置要求极低：

- **必需**：一台可以上网的电脑（Windows/macOS/Linux均可）
- **可选**：高中数学基础（学习深度学习部分时会用到）
- **可选**：基本的英语阅读能力（查阅文档时会用到）

无需任何编程经验，本教程从零开始讲解。

---

## 如何使用本教程

1. **按需选择章节**：根据自身基础和学习目标，选择合适的学习路径
2. **动手敲代码**：每个代码示例都要亲手输入并运行，不要复制粘贴
3. **理解后再继续**：确保理解当前内容后再学习下一节
4. **完成练习题**：每章末尾有练习题，建议独立完成后再查看答案
5. **实践项目**：学完每个部分后，尝试做一个小项目巩固知识

---

## 教程特色

- **27章完整内容**：从环境搭建到模型部署、异步并发与复杂系统编排
- **深度学习融合**：每章配有AI应用案例和PyTorch代码示例
- **135道练习题**：每章5道精选习题，含详细解答
- **零基础友好**：无需任何编程经验，从零开始讲解
- **中文编写**：专为中文学习者设计，术语准确，表达清晰

---

## 环境配置

本教程的代码示例使用以下环境：

```python
# 推荐环境
Python >= 3.9
PyTorch >= 2.0
NumPy >= 1.20
Pandas >= 1.4
Matplotlib >= 3.5
```

快速安装：
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install torch numpy pandas matplotlib seaborn jupyter
```

---

## 许可证

本项目采用 MIT 许可证开源。你可以自由地使用、复制、修改和分发本教程的内容。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*
