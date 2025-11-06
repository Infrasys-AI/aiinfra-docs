<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# CODE03: 大模型 Qwen3 蒸馏(DONE)

> Author by:汪袁烁、ZOMI

模型蒸馏（Knowledge Distillation）是一种让小型学生模型（Student Model）学习大型教师模型（Teacher Model）的知识和行为的技术，旨在让小模型以更少的参数实现接近大模型的性能。

本次实验使用 Qwen3-4B 作为教师模型，指导 Qwen3-0.6B 学生模型进行训练。通过蒸馏，我们希望 Qwen3-0.6B 能在特定任务（如数学推理、代码生成）上获得接近 Qwen3-4B 的表现，同时保持较小的参数规模和计算开销。

## 1. 环境准备

首先安装必要的库：PyTorch、Transformers、Hugging Face Hub 和 Datasets。以下代码块用于设置环境：

```python
# 安装依赖库
!pip install torch transformers huggingface_hub datasets

# 导入所需模块
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,7"
```

    Requirement already satisfied: torch in /home/yswang/miniforge3/lib/python3.12/site-packages (2.7.1)
    Requirement already satisfied: transformers in /home/yswang/miniforge3/lib/python3.12/site-packages (4.56.1)
    Requirement already satisfied: huggingface_hub in /home/yswang/miniforge3/lib/python3.12/site-packages (0.34.4)
    Requirement already satisfied: datasets in /home/yswang/miniforge3/lib/python3.12/site-packages (4.0.0)
    Requirement already satisfied: filelock in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (3.19.1)
    Requirement already satisfied: typing-extensions>=4.10.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (4.14.1)
    Requirement already satisfied: setuptools in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (79.0.1)
    Requirement already satisfied: sympy>=1.13.3 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (1.14.0)
    Requirement already satisfied: networkx in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (3.5)
    Requirement already satisfied: jinja2 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (3.1.6)
    Requirement already satisfied: fsspec in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (2025.3.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.5.1.17 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (9.5.1.17)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.3 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (0.6.3)
    Requirement already satisfied: nvidia-nccl-cu12==2.26.2 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (2.26.2)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (1.11.1.6)
    Requirement already satisfied: triton==3.3.1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from torch) (3.3.1)
    Requirement already satisfied: numpy>=1.17 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (2.2.6)
    Requirement already satisfied: packaging>=20.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (25.0)
    Requirement already satisfied: pyyaml>=5.1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (2025.9.1)
    Requirement already satisfied: requests in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (2.32.4)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (0.22.0)
    Requirement already satisfied: safetensors>=0.4.3 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (0.6.2)
    Requirement already satisfied: tqdm>=4.27 in /home/yswang/miniforge3/lib/python3.12/site-packages (from transformers) (4.67.1)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /home/yswang/miniforge3/lib/python3.12/site-packages (from huggingface_hub) (1.1.9)
    Requirement already satisfied: pyarrow>=15.0.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from datasets) (21.0.0)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /home/yswang/miniforge3/lib/python3.12/site-packages (from datasets) (2.3.2)
    Requirement already satisfied: xxhash in /home/yswang/miniforge3/lib/python3.12/site-packages (from datasets) (3.5.0)
    Requirement already satisfied: multiprocess<0.70.17 in /home/yswang/miniforge3/lib/python3.12/site-packages (from datasets) (0.70.16)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (3.12.15)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (1.4.0)
    Requirement already satisfied: attrs>=17.3.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (6.6.4)
    Requirement already satisfied: propcache>=0.2.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (0.3.2)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (1.20.1)
    Requirement already satisfied: idna>=2.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from yarl<2.0,>=1.17.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.3.0,>=2023.1.0->datasets) (3.10)
    Requirement already satisfied: charset_normalizer<4,>=2 in /home/yswang/miniforge3/lib/python3.12/site-packages (from requests->transformers) (3.4.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from requests->transformers) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /home/yswang/miniforge3/lib/python3.12/site-packages (from requests->transformers) (2025.8.3)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from sympy>=1.13.3->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/yswang/miniforge3/lib/python3.12/site-packages (from jinja2->torch) (3.0.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /home/yswang/miniforge3/lib/python3.12/site-packages (from pandas->datasets) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /home/yswang/miniforge3/lib/python3.12/site-packages (from pandas->datasets) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /home/yswang/miniforge3/lib/python3.12/site-packages (from pandas->datasets) (2025.2)
    Requirement already satisfied: six>=1.5 in /home/yswang/miniforge3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.17.0)


    /home/yswang/miniforge3/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

## 2. 蒸馏的核心思想

模型蒸馏的目的是将教师模型（Teacher）的知识“转移”到学生模型（Student）中。这里的关键在于**软标签（Soft Targets）**：教师模型输出的概率分布比原始数据的硬标签包含更多信息，例如类别间的相似性（即“暗知识”）。蒸馏通过最小化学生模型与教师模型输出的差异来实现知识转移。

蒸馏通常结合两种损失：

1.  **蒸馏损失（Distillation Loss）**：使用 KL 散度（Kullback-Leibler Divergence）衡量学生模型与教师模型输出的概率分布差异。
2.  **学生损失（Student Loss）**：学生模型与真实标签的交叉熵损失。

总损失是两者的加权和：  

$$
\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{KL} + (1 - \alpha) \cdot \mathcal{L}_{CE}
$$  

其中 $\alpha$ 是权重系数（通常设为 0.5-0.7），$\mathcal{L}_{KL}$ 是 KL 散度损失，$\mathcal{L}_{CE}$ 是交叉熵损失。

在 Softmax 函数中引入温度 $T$ 可以平滑概率分布：  

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$  

更高的 $T$ 值会使分布更平滑，揭示更多类别间关系。

## 3. 数据准备

我们使用简单的指令跟随数据集进行演示（如数学问题或代码生成任务）。这里以 `timdettmers/openassistant-guanaco` 数据集为例（包含指令-响应对）：

```python
# 加载数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")


def preprocess_function(examples):
    prompts = []
    for txt in examples["text"]:
        # 因为该 dataset 格式是 “### Human:” 和 “### Assistant:”
        if "### Assistant:" in txt:
            human_part, assistant_part = txt.split("### Assistant:", 1)
        else:
            human_part = txt
            assistant_part = ""
        human_part = human_part.strip()
        assistant_part = assistant_part.strip()
        # 拼成 prompt 形式：Human + Assistant
        prompt = human_part + "\n### Assistant: " + assistant_part
        prompts.append(prompt)
    return {"text": prompts}


# 选取子集以简化实验（500 条样本）
small_dataset = dataset.select(range(500)).map(preprocess_function, batched=True)
```

    Repo card metadata block was not found. Setting CardData to empty.


`load_dataset` 从 Hugging Face 加载数据集。`preprocess_function` 将指令和响应格式化为模型输入（例如："Instruction: What is 2+2?\nResponse: 4"）。

## 4. 教师和学生模型

使用 Hugging Face 的 `AutoModelForCausalLM` 加载 Qwen3-4B（教师）和 Qwen3-0.6B（学生）：


```python

# 定义模型名称
teacher_model_name = "Qwen/Qwen3-4B"
student_model_name = "Qwen/Qwen3-0.6B"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌

# 加载教师模型（使用 float16 节省显存）
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name, device_map="auto"
)

# 加载学生模型（同样使用 float16）
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name, device_map="auto"
)
```

    Loading checkpoint shards: 100%|██████████| 3/3 [00:03<00:00,  1.30s/it]


其中，`device_map="cuda:0"`将模型分配到 0 号 GPU 上，你也可以使用 `device_map="auto"` 自动将模型分配到可用设备（GPU/CPU）。你也可以使用`torch.float16` 减少显存占用，但可能略微影响精度（蒸馏中可接受）。我这里使用 FP32 加载用于后续的 AMP 训练。分词器使用教师模型的版本，确保输入处理一致。

## 5. 定义蒸馏损失函数

### 常规的蒸馏损失函数

我们需要自定义损失函数，结合 KL 散度和交叉熵损失：


```python
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=5.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 计算蒸馏损失（KL 散度）
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 计算学生损失（交叉熵）
        ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        # 结合损失
        return self.alpha * kl_loss + (1 - self.alpha) * ce_loss
```

这里采用了一种经典的软标签 + 硬标签混合蒸馏（soft-label distillation）方法，通过混合了软标签损失（KL 散度）和硬标签损失（交叉熵）来兼顾“模仿教师”与“符合真实标签”这两个目标。

其中，

- `alpha` 控制蒸馏损失与交叉熵损失的权重。
- `temperature` 平滑概率分布（更高值使教师输出更柔和）。
- `kl_loss` 计算学生与教师软标签的 KL 散度。
- `ce_loss` 计算学生输出与真实标签的交叉熵。
- 

### 尝试解决 OOM - 一种分 chunk 的蒸馏损失函数

由于直接算整个 vocab 的 DistillationLoss 容易导致 OOM，因此我们自然的会想到一种替代的方法。也即沿着最后一个维度（vocab）切分成多个 chunk，并且在最后拼接回去：


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# chunk 方法，但是分 chunk 可能导致 softmax 不能更好的捕捉整体
class DistillationLossWithChunk(nn.Module):
    def __init__(self, alpha=0.7, temperature=5.0, pad_id: int = None, num_chunks: int = 4):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.num_chunks = num_chunks
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        if pad_id is not None:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 把 logits 转为 float32 提高稳定性，我这里已经是 FP32 加载的了
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        T = self.temperature
        # logits 的最后一个维度是类别维度 (vocab size)
        vocab_size = student_logits.size(-1)

        # 将类别维度按 num_chunks 切分
        # chunks 是列表，每个 chunk 的形状 [*, chunk_size]
        student_chunks = torch.chunk(student_logits, self.num_chunks, dim=-1)
        teacher_chunks = torch.chunk(teacher_logits, self.num_chunks, dim=-1)

        total_kl = 0.0
        # 对每个 chunk 计算 KL
        for s_chunk, t_chunk in zip(student_chunks, teacher_chunks):
            # s_chunk 和 t_chunk 都是最后维度 = chunk_size
            # 做 softmax / logsoftmax
            # 注意 /T 缩放
            s_scaled = s_chunk / T
            t_scaled = t_chunk / T

            # soft teacher (概率分布)
            soft_t = torch.softmax(t_scaled, dim=-1)
            # log soft student
            log_s = torch.log_softmax(s_scaled, dim=-1)
            # KL for this chunk
            kl_chunk = self.kl_loss(log_s, soft_t) * (T * T)

            # 因为我们切块了类别维度，要做加权合并
            # 简单地平均或按块大小加权
            total_kl += kl_chunk * (s_chunk.size(-1) / vocab_size)

        # 硬标签交叉熵
        ce = self.ce_loss(
            student_logits.view(-1, vocab_size),
            labels.view(-1)
        )

        loss = self.alpha * total_kl + (1.0 - self.alpha) * ce
        return loss

```

但是值得注意的是，如果你使用这个作为蒸馏损失函数。整 vocab 上的 softmax / log_softmax + KL 可能被 chunk 分块插入时破坏，因为是对每个子块独立算 SoftMax：
$
\sum_{j=1}^V \exp\bigl(z_j / T\bigr)
$
再累加而不是对于整个 vocab 计算，因此会产生一定的偏差。

如果你对于 CUDA 足够了解，你可能会想“自己设计一个 fused kernel
把这些操作融合成一个 Kernel 不就能节约缓存了吗？”。是的，你可以基于 Liger Kernel 的 FusedLinearCrossEntropy（融合线性 + 交叉熵损失 + softmax/归一化）这种融合操作设计我们的 softmax + KL + CE 的蒸馏损失。当然由于这种方法并不是训练压缩的常规方法，而且需要支持中间激活值的保留，不具备什么可扩展性，因此我并不建议这种做法。我们在解决问题的时候可以更多时候学会借鉴前人的所作所为，这也是学习重要的一环。


### TopK 的方法

那么工业界和学术界往往如何处理这种 OOM 的训练压缩问题呢，我这里参考了[logits 的 topk 截断](https://arxiv.org/html/2410.16215v1)的方法,把 teacher_logits 截断为 top-k 。只保留比较重要的高概率的
知识以来节约显存开销：


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLossWithTopK(nn.Module):
    def __init__(self, alpha=0.7, temperature=5.0, pad_id: int = None, topk: int = None):
        """
        alpha, temperature 如常用于软标签 + 硬标签融合  
        pad_id 用于交叉熵时忽略 padding token  
        topk: 如果指定且 < vocab_size，则对 teacher_logits 做 top-k 截断
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.topk = topk

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        if pad_id is not None:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        假设形状如下：
        - student_logits, teacher_logits: [B, seq_len, V]
        - labels: [B, seq_len]
        """
        # 转为 float 以保证稳定性
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()

        T = self.temperature
        B, S, V = teacher_logits.size()

        # ========== top-k 截断 teacher_logits ==========

        if (self.topk is not None) and (self.topk < V):
            # flatten 前两维以方便 topk 操作
            flat_teacher = teacher_logits.view(B * S, V)  # [B*S, V]
            flat_student = student_logits.view(B * S, V)

            # topk 值与索引
            topk_vals, topk_idx = torch.topk(flat_teacher, self.topk, dim=-1)  # [B*S, topk]

            # mask 非 top-k 为 -inf
            mask = torch.full_like(flat_teacher, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)
            teacher_logits_trunc = mask.view(B, S, V)
            student_logits_trunc = flat_student.view(B, S, V)
        else:
            teacher_logits_trunc = teacher_logits
            student_logits_trunc = student_logits

        # ========== 缩放 / 温度处理 ==========

        t_scaled = teacher_logits_trunc / T
        s_scaled = student_logits_trunc / T

        # soft teacher & log soft student（只对截断后的 logits 计算 softmax / log-softmax）
        soft_teacher = torch.softmax(t_scaled, dim=-1)
        log_student = torch.log_softmax(s_scaled, dim=-1)

        kl = self.kl_loss(log_student, soft_teacher) * (T * T)

        # 硬标签交叉熵，用原始 student_logits（非缩放版）
        ce = self.ce_loss(
            student_logits.view(-1, V),
            labels.view(-1)
        )

        loss = self.alpha * kl + (1.0 - self.alpha) * ce
        return loss

```

可见，由于 student_logits 和 teacher_logits 通常张量很大，因此这种 topk 截断可以很好的保留关键信息。

## 6. 微调蒸馏循环

下面实现蒸馏训练循环（简化版），我这里选用了 TopK 的方法作为蒸馏损失函数：

```python
from torch.cuda.amp import autocast, GradScaler

# ===== 优化器和损失 =====
optimizer = optim.AdamW(student_model.parameters(), lr=5e-5)
distill_loss_fn = DistillationLossWithTopK(alpha=0.7, temperature=5.0, pad_id=tokenizer.pad_token_id)

# ===== AMP 相关 =====
scaler = GradScaler()  # 自动混合精度缩放器

# ===== 训练参数 =====
epochs = 3
batch_size = 2

# ===== 训练循环 =====
for epoch in range(epochs):
    student_model.train()
    total_loss = 0.0

    for i in range(0, len(small_dataset), batch_size):
        # 准备批量数据
        batch_texts = small_dataset["text"][i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(student_model.device) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()

        optimizer.zero_grad()

        # 教师模型推理（禁用梯度，float32 保证数值稳定）
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=False)
            teacher_logits = teacher_outputs.logits.float()

        # 学生前向 + 损失 (使用 autocast)
        with autocast():
            student_outputs = student_model(**inputs, labels=None)
            student_logits = student_outputs.logits
            loss = distill_loss_fn(student_logits, teacher_logits, labels)

        # 反向传播（自动缩放）
        scaler.scale(loss).backward()

        # 梯度裁剪（防止梯度爆炸）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

        # 参数更新
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Average Loss: {total_loss / (len(small_dataset)/batch_size):.4f}")

```

    /home/yswang/ncu_tmp/ipykernel_1728809/1586029240.py:8: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
      scaler = GradScaler()  # 自动混合精度缩放器
    /home/yswang/ncu_tmp/ipykernel_1728809/1586029240.py:34: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
      with autocast():


    Epoch 1, Average Loss: 595.2083
    Epoch 2, Average Loss: 569.3986
    Epoch 3, Average Loss: 548.7215

教师模型在推理时禁用梯度（`torch.no_grad()`），以减少计算和显存开销。使用小批量（`batch_size=4`）适应有限显存。损失函数同时考虑教师输出（软标签）和真实标签。
此外，使用了 AMP（自动混合精度 / Automatic Mixed Precision）方法，降低显存占用的同时加速了训练过程。

## 7. 评估蒸馏效果

训练后，我们在测试集上比较学生模型与教师模型的性能。使用简单的准确率（Accuracy）或困惑度（Perplexity）作为指标：

```python
# 评估函数
def evaluate_model(model, test_data):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for text in test_data["text"]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
            labels = inputs["input_ids"]
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()
    perplexity = torch.exp(torch.tensor(total_loss / len(test_data))).item()
    return perplexity

# 加载测试数据
test_dataset = load_dataset("timdettmers/openassistant-guanaco", split="test").select(range(100))

# 计算教师和学生的困惑度
teacher_ppl = evaluate_model(teacher_model, test_dataset)
student_ppl = evaluate_model(student_model, test_dataset)

print(f"Teacher Perplexity: {teacher_ppl:.2f}")
print(f"Student Perplexity: {student_ppl:.2f}")
```

    Repo card metadata block was not found. Setting CardData to empty.
    Using the latest cached version of the dataset since timdettmers/openassistant-guanaco couldn't be found on the Hugging Face Hub
    Found the latest cached dataset configuration 'default' at /home/yswang/.cache/huggingface/datasets/timdettmers___openassistant-guanaco/default/0.0.0/831dabac2283d99420cda0b673d7a2a43849f17a (last modified on Sat Oct  4 14:17:06 2025).


    Teacher Perplexity: 6.97
    Student Perplexity: 32.28

**困惑度（Perplexity）** 衡量模型预测能力（越低越好）。蒸馏后，学生模型的困惑度应接近教师模型。实际应用中还可使用任务特定指标（如数学问题的准确率）。

## 8. 总结与思考

在本实验中，我们期望蒸馏后的 Qwen3-0.6B 性能显著提升。例如，在测试集上，学生模型的困惑度可能从原始值（例如 30+）降低到接近教师模型的水平（例如 15-20）。然而，蒸馏效果受多种因素影响：

1.  **数据质量**：高质量、多样化的数据能提升蒸馏效果。Qwen3 预训练数据涵盖多语言和多种领域（如代码、数学），这有助于蒸馏。
2.  **超参数选择**：温度参数 $\alpha$ 和 $T$ 需要调优。过高的 $T$ 可能使分布过于平滑，而过低的 $\alpha$ 可能忽略教师知识。
3.  **模型容量差距**：学生模型过小可能无法完全吸收教师知识（Qwen3-0.6B 与 Qwen3-4B 的参数量比约为 1:6.7，差距适中）。
