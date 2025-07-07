# 单机单卡 MoE

解读 by [AI 布道 Mr.Jin]

其实在 DeepSeek-R1 爆火之前，DeepSeek V2 在我们行业就已经妇孺皆知了，它独特的 MOE 结构值得研究一下。这篇文章是基于 ZOMI 酱 的这个视频写的：《使用昇腾 NPU 手撕 MoE 单机版代码！没想到如此简单！》。

通过《09MOECore 解读》，我们知道了 MOE 的结构原理是什么样的，接下来看一下代码上是怎么实现的！

## MOE 计算代码

下面是 zomi 酱课程中提供的完整代码：


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        
        # 路由网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 专家集合
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        
    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device
        
        # 路由计算
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # 辅助损失计算
        if self.training:
            # 重要性损失（专家利用率均衡）
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)
            
            # 负载均衡损失（样本分配均衡）
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()
            
            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0.0

        # 专家分配逻辑
        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()

        # 初始化输出
        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features, 
                            device=device)

        # 处理每个专家
        for expert_idx in range(self.num_experts):
            # 获取分配给当前专家的样本
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            # 容量控制
            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue

            # 处理专家计算
            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            
            # 累加输出
            outputs.index_add_(0, expert_samples, weighted_output)

        return outputs, aux_loss

# 测试示例
if __name__ == "__main__":
    input_dim = 5
    output_dim = 10
    num_experts = 8
    top_k = 3
    expert_capacity = 32
    hidden_dim = 512
    batch_size = 10

    # add
    device = torch.device("cpu")
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)
    moe.eval()
    output, _ = moe(x)
    print(f"Eval output shape: {output.shape}") # torch.Size([64, 256])
```

    C:\Users\Administrator\jupyter-env\lib\site-packages\torch\_subclasses\functional_tensor.py:276: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:81.)
      cpu = _conversion_method_template(device=torch.device("cpu"))
    

    Eval output shape: torch.Size([10, 10])
    

接下来，我们把每一部分拆解进行解读。

### 初始化函数定义

首先，定义了 Expert 类，也就是“专家”，可以看到，专家是由线性层和激活函数构成的简单模型。

然后开始定义 MOE 类。在初始化函数中，定义了这样几个变量：

self.num_experts：专家的数量，也就是上面提到的“并列线性层”的个数，训练后的每个专家的权重都是不同的，代表它们所掌握的“知识”是不同的。

self.top_k：每个输入 token 激活的专家数量。

self.expert_capacity：代表计算每组 token 时，每个专家能被选择的最多次数。

self.gate：路由网络，一般是一个线性层，用来计算每个专家被选择的概率。

self.experts：实例化 Expert 类，生成多个专家。


```python
num_experts = num_experts
top_k = top_k
expert_capacity = expert_capacity
# 路由网络
gate = nn.Linear(input_dim, num_experts)
# 专家集合
experts = nn.ModuleList(
    [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
```

### 前向计算逻辑

接下来看一下 forward 函数。

首先是输入 x，shape 是（batch_size, input_dim），batch_size 我们可以看作是 token 的数量，也就是序列长度。然后通过 self.gate 和 softmax 计算每个 token 在每个专家上的激活概率：


```python
logits = gate(x)
probs = torch.softmax(logits, dim=-1)
print("probs: ", probs)
```

    probs:  tensor([[0.1105, 0.0906, 0.1629, 0.1508, 0.2257, 0.1269, 0.0388, 0.0938],
            [0.0668, 0.1061, 0.0902, 0.1864, 0.2158, 0.1080, 0.0913, 0.1354],
            [0.0482, 0.0661, 0.0373, 0.1738, 0.2768, 0.0696, 0.1436, 0.1845],
            [0.1450, 0.0297, 0.0412, 0.1784, 0.2312, 0.1261, 0.0879, 0.1605],
            [0.2216, 0.0650, 0.0464, 0.0996, 0.0547, 0.3725, 0.0915, 0.0487],
            [0.1987, 0.0730, 0.1046, 0.0963, 0.0684, 0.3503, 0.0533, 0.0553],
            [0.0512, 0.1033, 0.0112, 0.2495, 0.0582, 0.1068, 0.3491, 0.0707],
            [0.1033, 0.1161, 0.0553, 0.2258, 0.1429, 0.1449, 0.1225, 0.0892],
            [0.0377, 0.1224, 0.1002, 0.1947, 0.2121, 0.0792, 0.0942, 0.1596],
            [0.0441, 0.1337, 0.0439, 0.1240, 0.1968, 0.1091, 0.2043, 0.1441]],
           grad_fn=<SoftmaxBackward0>)
    

probs 的打印结果如上：我们设置的 batch_size 是 10，num_experts 是 8，所以 probs 是个 10 行 8 列的矩阵。

接着，再用 topk 算子把每个 token 的激活专家选出来：


```python
topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
print("topk_probs: ", topk_probs)
print("topk_indices: ", topk_indices)
```

    topk_probs:  tensor([[0.2257, 0.1629, 0.1508],
            [0.2158, 0.1864, 0.1354],
            [0.2768, 0.1845, 0.1738],
            [0.2312, 0.1784, 0.1605],
            [0.3725, 0.2216, 0.0996],
            [0.3503, 0.1987, 0.1046],
            [0.3491, 0.2495, 0.1068],
            [0.2258, 0.1449, 0.1429],
            [0.2121, 0.1947, 0.1596],
            [0.2043, 0.1968, 0.1441]], grad_fn=<TopkBackward0>)
    topk_indices:  tensor([[4, 2, 3],
            [4, 3, 7],
            [4, 7, 3],
            [4, 3, 7],
            [5, 0, 3],
            [5, 0, 2],
            [6, 3, 5],
            [3, 5, 4],
            [4, 3, 7],
            [6, 4, 7]])
    

topk_probs 和 topk_indices 的打印结果如上，因为我们设置的 top_k=3，所以每个 token 都把排名前三的概率选出来了，同时 topk_indices 把这些概率对应的专家编号也选出来了。

self.training 分支对应的是训练过程中计算损失函数的部分，我们后面再讲。

选择好专家后，就要开始计算了。计算规则是，对于每个 token，假如它选择的专家是 e1、e2、e3，概率分别是 p1、p2、p3，那么这个 token 的计算结果就是 p1xe1_out+p2xe2_out+p3xe3_out。

由于计算个体是每个专家，所以代码中用 for 循环遍历每个专家。我们以第 0 个专家为例，看看它的计算过程是怎样的。

首先需要确定 0 号专家的输入。由于不是每个 token 都选择了 0 号专家，所以不能把 x 直接作为输入，而是要确定一个下标向量 idxes，把 x[idxes]作为 0 号专家的输入，idxes 的值就是激活了 0 号专家的所有 token 编号，那么怎么得到 idxes 呢？代码里面是这样做的：

首先计算一个 mask（假设 expert_idx=0）：


```python
flat_indices = topk_indices.view(-1)
expert_idx = 0
expert_mask = flat_indices == expert_idx
print(expert_mask)
```

    tensor([False, False, False, False, False, False, False, False, False, False,
            False, False, False,  True, False, False,  True, False, False, False,
            False, False, False, False, False, False, False, False, False, False])
    

flat_indices 是 topk_indices 平铺之后的向量。通过对比，可以看到 expert_mask 中 True 的位置和 topk_indices 中 0 的位置铺平之后是一致的，代表第 0 个专家被第 4 个和第 5 个 token 激活了。

而且 expert_mask 代表的含义是：只要它的第 0-2 的位置是 True 的话，就代表被第 0 个 token 激活了，只要它的第 3-5 的位置是 True 的话，就代表被第 1 个 token 激活了，以此类推，我们可以声明一个 sample_indices 向量：


```python
sample_indices = torch.arange(batch_size, device=device)[:, None].expand(-1, top_k).flatten()
print(sample_indices)
```

    tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9])
    

再通过下面的代码就可以把 idxes 取出来了：


```python
expert_samples = sample_indices[expert_mask]
print(expert_samples)
```

    tensor([4, 5])
    

也顺便把概率权重取出来：


```python
flat_probs = topk_probs.view(-1)
expert_weights = flat_probs[expert_mask]
print(expert_weights)
```

    tensor([0.2216, 0.1987], grad_fn=<IndexBackward0>)
    

接着把输入取出来：


```python
expert_input = x[expert_samples]
print(expert_input)
```

    tensor([[-0.7454, -1.4269, -1.1833, -0.2611,  1.4887],
            [-0.9482, -1.9723, -0.2507,  0.4739,  1.0142]])
    

再进行专家计算：


```python
expert_output = experts[expert_idx](expert_input)
weighted_output = expert_output * expert_weights.unsqueeze(-1)
```

最后还需要把计算结果叠加到对应的 token 上面去：


```python
outputs = torch.zeros(batch_size, experts[0].net[-1].out_features)
outputs.index_add_(0, expert_samples, weighted_output)
```




    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0145, -0.0041,  0.0212, -0.0435, -0.0893, -0.0695, -0.0123, -0.0462,
             -0.1006,  0.0255],
            [-0.0135,  0.0327,  0.0766, -0.0206, -0.0717, -0.0597, -0.0358, -0.0326,
             -0.0773,  0.0074],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000]], grad_fn=<IndexAddBackward0>)



完成上面的 for 循环之后，就把所有专家的计算任务完成了，通过 index_add_的操作，把每个 token 的计算结果也汇总了。

### 损失函数

损失函数包含 2 部分：专家利用率均衡和样本分配均衡。

首先是专家利用率均衡，如果每个专家被选择的概率相近，那么说明分配越均衡，损失函数越小：

```
importance = probs.sum(0)
importance_loss = torch.var(importance) / (self.num_experts ** 2)
```

然后是样本分配均衡，首先得到每个 token、每个专家的分配概率矩阵：

```
mask = torch.zeros_like(probs, dtype=torch.bool)
mask.scatter_(1, topk_indices, True)
routing_probs = probs * mask
```

然后按照 token 维度（样本维度）求平均，得到每个专家被分配的 token 平均数量和平均概率：

```
expert_usage = mask.float().mean(0)
routing_weights = routing_probs.mean
```

两者相乘求和得到负载均衡损失：

```
load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()
```

样本分配越均衡，这个损失函数越小。举个例子，10 个专家，10 个样本，如果所有样本都分到 1 个专家，那么损失函数值为 10x1+0+0...+0=10，如果平均分给 10 个专家，那么损失函数值为 1x0.1+1x0.1+...+1x0.1=1。

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?isOutside=true&aid=114133034867676&bvid=BV1UTRYYUE5o&cid=28782825014&p=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
