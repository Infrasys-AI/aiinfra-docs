# MOE 模型可视化

解读 by [AI 布道 Mr.Jin]

其实在 DeepSeek-R1 爆火之前，DeepSeek V2 在我们行业就已经妇孺皆知了，它独特的 MOE 结构值得研究一下。

## MOE 结构概述

我们可以从 zomi 酱视频里面的这张图开始：

![file](http://image.openwrite.cn/33719_D3965B8EB13E4241A927D74DA36C0837.png)

MOE 是 mixture of experts 的缩写，简单来说，就是把传统 transformer 结构中 decoder 层里面的单个线性层替换层多个并列的线性层。在这些线性层前面还有一个 Router，Router 会选择并列线性层里面的一部分进行计算。这样的话，既能让模型学习更多的知识（多个“专家”），又能减少推理计算量（选择部分“专家”进行计算）。接下来我们从 Router（也叫 Gate）模块、MOE 推理模块和损失函数模块这 3 个方面进行解读。

## Router 模块

Router 模块的输入是序列特征，形状是[batch_size, seq_len, hidden_dim]，输出是 select_expert_id 和 expert_weight，shape 都是[batch_size, seq_len, topk]，topk 是为每个 token 选择的专家数量。

Router 模块实际上是由全连接层、softmax 层以及 topk 算子组成，如果全部候选专家的数量一共是 expert_num，那么全连接层的输出 shape 是[batch_size, seq_len, expert_num]，代表每个 token 被分配到每个候选专家的概率，然后使用 softmax 对概率值进行归一化，最后使用 topk 算子把概率排在前面的专家选择出来，得到的输出 shape 就是[batch_size, seq_len, topk]。

举个例子，假如 batch_size=1, seq_len=5, expert_num=6, topk=3，那么 Router 模块中的 topk 最后输出可能是[[0, 1, 2], [2, 4, 5], [1, 2,  3], [0, 3, 5], [3, 4, 5]]和[[0.2, 0.3, 0.3], [0.25, 0.28, 0.32], [...], [...], [...]]。这个输出代表第 1 个 token 会给 0 号专家、1 号专家和 2 号专家计算，然后在推理模块中会把他们的结果分别乘以 0.2、0.3、0.3 的权重，第 2 个 token 会给 2 号专家、4 号专家和 5 号专家计算，然后在推理模块中会把他们的结果分别乘以 0.25、0.28、0.32 的权重，以此类推。所以，Router 的功能就是把不同的 token 分给不同的 expert，这也是它为什么叫“路由”的原因。

## MOE 推理模块

完成路由之后，每个专家就要开始计算了。每个专家需要收集自己负责计算的 token，还是以上面给的例子为例，0 号专家负责第 1 个 token 和第 4 个 token 的计算，所以 0 号专家的输入 shape 是[2, hidden_dim]；1 号专家负责第 1 个 token 和第 3 个 token 的计算，所以 0 号专家的输入 shape 是[2, hidden_dim]，以此类推。

各个专家完成计算后，我们又要把计算进行组合，得到每个 token 的推理结果。继续上面的例子，假如第 1 个 token 在 0 号专家、1 号专家和 2 号专家的计算结果分别为 result_0, result_1, result2，那么整个 MOE 模块对第 1 个 token 的预测结果就是 result_0x0.2+result_1x0.3+result2x0.3。

## 损失函数模块

损失函数包含 2 部分：专家利用率均衡和样本分配均衡。

专家利用率均衡的计算公式是 var(prob_list)，也就是所有专家被选择的概率之和的方差。如果每个专家被选择的概率相近，那么说明分配越均衡，整个系统的算力利用率就越高，否则会造成某些计算节点的闲置浪费。

然后是样本分配均衡，计算公式是 sum(token_num_list*prob_list)，也就是把各专家分配到的 token 数量列表和概率之和列表相乘求和。样本分配越均衡，这个损失函数越小。举个例子，10 个专家，10 个样本，如果所有样本都分到 1 个专家，那么损失函数值为 10x1+0+0...+0=10，如果平均分给 10 个专家，那么损失函数值为 1x0.1+1x0.1+...+1x0.1=1。

## 本节视频

<html>
<iframe src="https://player.bilibili.com/player.html?isOutside=true&aid=114109882308562&bvid=BV1Gj9ZYdE4N&cid=28706275882&p=1&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
