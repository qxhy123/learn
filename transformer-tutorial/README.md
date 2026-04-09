# 从零到高阶的Transformer教程

## 项目简介

本教程旨在为学习者提供一套系统、完整的Transformer学习资源，从注意力机制的数学原理出发，循序渐进地覆盖Transformer架构、预训练模型、微调技术等核心主题，最终深入到大语言模型的工程实践。

**本教程的独特之处**：每章都包含从零实现的PyTorch代码，不依赖高层封装，帮助读者真正理解每个组件的工作原理。

---

## 目标受众

- 有Python和深度学习基础的学习者
- 希望深入理解Transformer原理的研究者
- 准备开发NLP/CV应用的工程师
- 对大语言模型原理感兴趣的开发者
- 大学AI相关专业的在校学生

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：数学基础

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第1章 | [序列建模基础](./part1-foundations/01-sequence-modeling.md) | RNN/LSTM回顾、序列到序列、长距离依赖 | RNN梯度消失可视化 |
| 第2章 | [注意力机制原理](./part1-foundations/02-attention-basics.md) | 注意力直觉、点积注意力、缩放点积 | 注意力权重可视化 |
| 第3章 | [位置编码](./part1-foundations/03-positional-encoding.md) | 正弦编码、可学习编码、相对位置编码 | 位置编码实现与可视化 |

### 第二部分：注意力机制深入

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第4章 | [自注意力机制](./part2-attention/04-self-attention.md) | Query-Key-Value、自注意力计算、复杂度分析 | 自注意力从零实现 |
| 第5章 | [多头注意力](./part2-attention/05-multi-head-attention.md) | 多头机制原理、并行计算、头分析 | 多头注意力实现 |
| 第6章 | [掩码注意力](./part2-attention/06-masked-attention.md) | Padding掩码、因果掩码、交叉注意力掩码 | 各类掩码实现 |

### 第三部分：Transformer架构

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第7章 | [编码器结构](./part3-architecture/07-encoder.md) | 层归一化、前馈网络、残差连接 | 编码器从零实现 |
| 第8章 | [解码器结构](./part3-architecture/08-decoder.md) | 自回归生成、编码器-解码器注意力 | 解码器从零实现 |
| 第9章 | [完整Transformer](./part3-architecture/09-full-transformer.md) | 输入嵌入、完整架构、端到端实现 | 完整Transformer实现 |

### 第四部分：训练技术

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第10章 | [训练策略](./part4-training/10-training-strategies.md) | 学习率Warmup、标签平滑、Dropout | 学习率调度器实现 |
| 第11章 | [优化技术](./part4-training/11-optimization.md) | Adam/AdamW、梯度累积、混合精度 | 训练循环优化 |
| 第12章 | [损失函数与评估](./part4-training/12-loss-evaluation.md) | 交叉熵、序列级损失、BLEU/ROUGE | 评估指标实现 |

### 第五部分：预训练模型

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第13章 | [BERT原理与实现](./part5-variants/13-bert.md) | 双向编码、MLM与NSP任务 | BERT从零实现 |
| 第14章 | [GPT系列](./part5-variants/14-gpt.md) | 自回归语言模型、GPT演进、ICL | GPT从零实现 |
| 第15章 | [T5与统一框架](./part5-variants/15-t5.md) | Text-to-Text、编码器-解码器预训练 | T5架构实现 |
| 第16章 | [现代大模型](./part5-variants/16-modern-llm.md) | LLaMA、RoPE、GQA、Flash Attention | 现代LLM组件实现 |

### 第六部分：微调与应用

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第17章 | [微调技术](./part6-applications/17-finetuning.md) | 全参数微调、冻结策略、学习率设置 | 微调实战 |
| 第18章 | [参数高效微调](./part6-applications/18-peft.md) | LoRA、Adapter、Prefix-Tuning | LoRA从零实现 |
| 第19章 | [下游任务实战](./part6-applications/19-downstream-tasks.md) | 文本分类、序列标注、问答、生成 | 四大任务实现 |

### 第七部分：高级主题

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第20章 | [高效Transformer](./part7-advanced/20-efficient-transformer.md) | 稀疏注意力、线性注意力、Longformer | 高效注意力实现 |
| 第21章 | [多模态Transformer](./part7-advanced/21-multimodal.md) | ViT、CLIP、多模态融合 | ViT从零实现 |
| 第22章 | [Transformer可解释性](./part7-advanced/22-interpretability.md) | 注意力可视化、探针任务、BertViz | 可解释性工具 |

### 第八部分：工程实践

| 章节 | 标题 | 主要内容 | 代码实战 |
|------|------|----------|----------|
| 第23章 | [推理优化](./part8-engineering/23-inference-optimization.md) | KV缓存、量化、蒸馏、ONNX | 推理优化实现 |
| 第24章 | [完整项目实战](./part8-engineering/24-complete-project.md) | 训练小型GPT、HF集成、部署 | 端到端项目 |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [数学符号速查](./appendix/math-reference.md) | 矩阵运算、softmax、层归一化公式 |
| 附录B | [PyTorch API参考](./appendix/pytorch-api.md) | nn.Transformer、Hugging Face API |
| 附录C | [练习答案汇总](./appendix/answers.md) | 各章练习题答案索引 |

---

## 学习路径建议

### 路径一：Transformer入门（约 4-5 周）

适合有深度学习基础、首次学习Transformer的学习者：

1. 系统学习第1-3章（数学基础）
2. 学习第4-6章（注意力机制）
3. 学习第7-9章（Transformer架构）
4. 完成每章的基础练习题

### 路径二：预训练模型深入（约 3-4 周）

适合已了解Transformer、希望深入预训练模型的学习者：

1. 快速复习第4-9章
2. 重点学习第13-16章（预训练模型）
3. 学习第17-19章（微调与应用）
4. 完成中级和提高题

### 路径三：LLM工程实践（约 2-3 周）

适合希望将Transformer应用于实际项目的工程师：

1. 复习第9章（完整Transformer）
2. 学习第16章（现代大模型）
3. 重点学习第18章（参数高效微调）
4. 实战第23-24章（工程实践）

---

## 前置要求

学习本教程需要以下基础：

- **必需**：Python编程基础
- **必需**：PyTorch基础（张量操作、nn.Module）
- **必需**：深度学习基础（反向传播、梯度下降）
- **推荐**：线性代数基础（矩阵乘法、转置）
- **推荐**：了解RNN/LSTM（第1章会复习）

---

## 环境配置

本教程的代码示例使用以下环境：

```bash
# 推荐环境
Python >= 3.9
PyTorch >= 2.0
transformers >= 4.30
numpy >= 1.20
matplotlib >= 3.5
```

快速安装：
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install torch torchvision torchaudio
pip install transformers datasets tokenizers
pip install numpy pandas matplotlib seaborn
pip install jupyter tqdm
```

---

## 教程特色

- **24章完整内容**：从注意力机制到完整项目部署
- **从零实现**：每个核心组件都有手写PyTorch实现
- **120道练习题**：每章5道精选习题，含详细解答
- **可视化丰富**：注意力热图、训练曲线、架构图
- **中文编写**：专为中文学习者设计，术语准确

---

## 许可证

本项目采用 MIT 许可证开源。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*
