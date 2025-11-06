# bitsandbytes LLM.int8()源码阅读

authored by:汪袁烁

## 调用的时候发生了什么
为了更深入的了解 INT8 量化中的 LLM.int8()，以及当我们调用：
```py
# 加载 INT8 精度的模型
model_int8 = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 启用 INT8 量化
    device_map="auto"
)
```
之后究竟发生了什么，我摘出了 LLM.int8()的源码来简要阅读，本篇适合那些对于 LLM.int8()量化想有更深入了解的同学。首先我们先准备一下源码：

```bash
#源码准备
git clone git@github.com:bitsandbytes-foundation/bitsandbytes.git
```
## bnb_linear 的替换

更具体的，当`Transformers` 在 `from_pretrained` 里检测到 8bit 量化后，会引入 `bitsandbytes` 集成代码，并把模型里的 `nn.Linear` 替换成 `bnb` 的 8bit 线性层：

```py
# 本段源码的位置：
# cd AIInfra/05Infer/06CompDistill/third-party/bitsandbytes/bitsandbytes/utils.py
import torch

def replace_linear(
    model: torch.nn.Module,
    linear_replacement: torch.nn.Module,
    skip_modules=("lm_head",),
    copy_weights: bool = False,
    post_processing_function: Optional[str] = None,
) -> torch.nn.Module:
    """
    遍历一个模型，把其中的 torch.nn.Linear 层替换为新的线性层。

    参数：
        model (`torch.nn.Module`):
            要处理的模型（或者子模块）；函数会递归遍历所有子模块。
        linear_replacement (`torch.nn.Module` or callable):
            用来替换原来 Linear 的新 Linear 类／构造函数。
            如果需要传额外参数，可传入 lambda 或者偏函数等方式封装。
        skip_modules (`tuple[str]`, 可选，默认为 ("lm_head",)):
            模块名字里如果包含这些字符串，就不替换它们。
            通常像 `lm_head`（语言模型的首输出线性层）这样的层是敏感的，
            替换可能导致输出维度／损失计算不一致／行为变差，所以默认跳过。
        copy_weights (`bool`):
            如果为 True，替换后的新线性层会把原来的权重 (weight) 和偏置 (bias) 从旧模块复制过来。
            如果为 False，就用新层的初始化权重／bias。
        post_processing_function (`str` 可选):
            替换后如果新模块需要做一些额外处理／初始化，可以传入这个名字。
            函数会尝试在旧模块上调用 `getattr(module, post_processing_function)`，
            如果存在就执行这个函数。这个参数用于比如某些 Linear 层创建时需要额外设置的情形。

    返回：
        返回替换后的模型（原地修改），所有不在 skip_modules 中的 Linear 层均被替换。

    """

    # 遍历 model 的直接子模块（named_children 提供“名字 + 子模块 对”）
    for name, module in model.named_children():
        # 如果子模块本身还有子模块（module.children 非空），递归调用替换函数
        # 这样可以深入所有层级（Transformer block、子 layer 等）
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights, post_processing_function)

        # 判断这个 module 是否是标准的 Linear 且名字不在 skip_modules 中
        if isinstance(module, torch.nn.Linear) and name not in skip_modules:

            # 保存旧的 Linear 模块（用于权重复制／bias 复制等）
            old_module = model._modules[name]

            # 用新的 Linear 替换它；linear_replacement 通常是一个 class 或构造函数／lambda
            # 这里按标准 Linear 的构造参数 in_features, out_features, bias 是否存在
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )

            # 如果 copy_weights 为 True，把旧模块的 weight 和 bias 复制到新模块里
            if copy_weights:
                # 要注意：这里直接把旧 module 的权重张量赋值给新 module 的 weight
                model._modules[name].weight = old_module.weight
                # 若有 bias
                if module.bias is not None:
                    model._modules[name].bias = old_module.bias

            # 如果指定了 post_processing_function，就取旧 module 上是否有这个函数
            # 如果有，就调用它。这样可以让一些需要额外初始化／配置的操作在替换后执行。
            if post_processing_function is not None:
                func = getattr(module, post_processing_function, None)
                if func is not None:
                    func(module)

    return model

```

## 生成 8bit 线性层参数

### 量化参数的准备 - Int8Params
在完成把模型里的 nn.Linear 自动替换成 bitsandbytes.nn.Linear8bitLt 之后，这些 8bit 线性层在上卡/前向时，把权重量化为 INT8，并生成 Int8Params 里的字段（如 CB/SCB/has_fp16_weights）：
```py
# 要阅读本段源码，你需要先：
# cd AIInfra/05Infer/06CompDistill/third-party/bitsandbytes/bitsandbytes/nn/modules.py
# 仅保留 CUDA 相关逻辑的精简版 Int8Params
import copy
from typing import Optional, Union, TypeVar
import torch
import bitsandbytes as bnb

T = TypeVar("T")
Tensor = torch.Tensor
dtype = torch.dtype
device = torch.device

class Int8Params(torch.nn.Parameter):
    """
    这里我只取 CUDA 了相关的函数：
    - 继承自 torch.nn.Parameter（实际底层是 torch.Tensor 的子类），
      用于承载量化后的权重以及量化所需的元数据（比如缩放因子）。
    - 设计目标：当参数被移动到 CUDA（例如 .to("cuda") 或 .cuda()）时，
      自动将 FP16/FP32 权重量化为 INT8，并把量化后的缓存（CB/SCB）挂在对象上。
      这种“上卡即量化”的做法与 bitsandbytes 的工作流契合。:contentReference[oaicite:1]{index=1}
    """

    def __new__(
        cls,
        data: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        has_fp16_weights: bool = False,
        CB: Optional[torch.Tensor] = None,
        SCB: Optional[torch.Tensor] = None,
    ):
        # __new__ 用来创建 *Tensor 子类实例*，此处若 data 为空就先给一个空张量。
        if data is None:
            data = torch.empty(0)

        # 关键：用 Tensor._make_subclass 建立“真正的”Tensor 子类实例，
        # 而不是普通的 Python 对象。这让 autograd/设备迁移等核心机制继续生效。
        # （_make_subclass/_make_wrapper_subclass 是 PyTorch 提供的底层接口，用于定制 Tensor 子类行为。）:contentReference[oaicite:2]{index=2}
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)

        # —— 量化相关的“伴随缓存” —— #
        # CB: 量化后的权重块（Compressed/Quantized Bits），通常是 int8 row-major 存储
        # SCB: scale（及可能的校正项），用于把 int8 乘法结果反量化回 FP16/FP32 域
        # 这两者是 bitsandbytes 的向量级（vector-wise）量化接口所返回的核心内容。:contentReference[oaicite:3]{index=3}
        obj.CB = CB
        obj.SCB = SCB

        # has_fp16_weights = True 时保留高精度权重（例如用于训练/混合精度或某些敏感路径）
        obj.has_fp16_weights = has_fp16_weights
        return obj

    def _quantize(self, cuda_dev: Union[int, device, str]):
        """
        在 *迁移到 CUDA* 时执行的量化步骤：
        - 若 has_fp16_weights=True，则直接把原始权重放到目标设备（保持 FP16/FP32，不做量化）。
        - 否则：把权重转成 FP16（更稳健的量化前置精度），调用
          bnb.functional.int8_vectorwise_quant 做“向量级”INT8 量化，得到 (CB, SCB) 并缓存。
          这种“每列/每向量一组 scale”的做法能更好适配大模型中的列间方差差异，是 LLM.int8 的关键之一。:contentReference[oaicite:4]{index=4}
        """
        if self.has_fp16_weights:
            # 保持高精度：直接把 data 搬到 CUDA（不量化）
            return super().to(cuda_dev)

        # contiguous(): 确保内存连续；to(..., dtype=torch.float16): 量化前先对齐到 FP16
        B = self.data.contiguous().to(device=cuda_dev, dtype=torch.float16)

        # int8_vectorwise_quant 会返回：
        #  - CB: int8 压缩后的权重块（row-major）
        #  - SCB: 对应的量化缩放/校正因子（用于反量化或混合精度合并）
        #  - 其它返回值（此处用下划线忽略）
        # 参考：bitsandbytes functional 文档、Lightning 中的实现注释。:contentReference[oaicite:5]{index=5}
        CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B)

        # 将当前参数的数据指向量化后的 int8 权重，并保存量化元数据
        self.data = CB
        self.CB = CB
        self.SCB = SCB
        return self

    def cuda(self, device: Optional[Union[int, device, str]] = None, non_blocking: bool = False):
        """
        便捷接口：与 .to("cuda") 等价，但更直观。
        这里直接转发到 .to(...)，让统一的迁移逻辑（含量化）生效。
        """
        return self.to(device=("cuda" if device is None else device), non_blocking=non_blocking)

    # 仅保留与 CUDA 相关的 to() 分支（去掉 CPU/XPU/其他平台逻辑，专注“上卡即量化”）
    def to(self, *args, **kwargs):
        # 解析目标设备/类型/非阻塞标志；PyTorch 内部接口用于实现 .to 的多样参数签名。
        cuda_dev, want_dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)

        # 仅当目标是 CUDA，且当前数据还在 CPU 时，执行“迁移并量化”。
        if cuda_dev is not None and isinstance(cuda_dev, torch.device) and cuda_dev.type == "cuda":
            if self.data.device.type == "cpu":
                # 上卡时触发 INT8 量化（CB/SCB 就地生成并缓存）
                return self._quantize(cuda_dev)

        # 其余情况：比如已经在 CUDA 上，仅需要 dtype 转换或复制，就走父类的 to，
        # 然后把我们的伴随缓存（CB/SCB、has_fp16_weights）转移到新参数对象上。
        new_param = Int8ParamsCUDA(
            super().to(device=cuda_dev, dtype=want_dtype, non_blocking=non_blocking),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
        )
        new_param.CB = self.CB
        new_param.SCB = self.SCB
        return new_param

    def __deepcopy__(self, memo):
        """
        深拷贝：确保 data/CB/SCB/has_fp16_weights 都被正确复制。
        这在做“模型拷贝/分发/保存中间状态”时很有用。
        """
        return type(self).__new__(
            type(self),
            data=copy.deepcopy(self.data, memo),
            requires_grad=self.requires_grad,
            has_fp16_weights=self.has_fp16_weights,
            CB=copy.deepcopy(self.CB, memo),
            SCB=copy.deepcopy(self.SCB, memo),
        )


```
代码很多，我们重点关注：
```py
CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B)
```
也是 INT8 量化的核心代码，当你`toCUDA`的时候，它会自动跳转到自己设计的 CUDA 内核进行 INT8 量化


### 量化的秘密 - int8_vectorwise_quant Kernel 实现
它中间辗转经过了一系列调用，我们直接定位到这个 Kernel 的实现：
```cpp
// 模板参数：
// T               —— 输入矩阵元素类型（通常是半/单精度，如 __half 或 float）。
// THREADS         —— 每个 block 启用的线程数（决定并行度和分条访问步幅）。
// SPARSE_DECOMP   —— 稀疏分解开关（编译期常量）：1 表示对“离群值(＞threshold)”不量化而置 0。
template <typename T, int THREADS, int SPARSE_DECOMP>

// __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
//   - 向编译器传达“本 kernel 以最多 1024 线程/块”为优化目标，并期望每个 SM 至少常驻 BNB_MAX_THREADS_PER_SM/1024 个 block。
//   - 编译器据此在“寄存器使用 vs. 并发(occupancy)”间做权衡，避免过多寄存器导致并发下降（或溢出到本地内存）。:contentReference[oaicite:0]{index=0}
__launch_bounds__(1024, BNB_MAX_THREADS_PER_SM / 1024) __global__
void kInt8VectorQuant(T* __restrict__ A,      // [rows, cols] 行主存放的输入矩阵
                      int8_t* out,            // [rows, cols] 行主存放的量化输出
                      float* rowStats,        // [rows] 逐行的统计量（这里存每行的 absmax）
                      float threshold,        // 稀疏分解中的离群值阈值
                      int rows, int cols) {

    // 对 Maxwell 架构(sm50/52) 且 CUDA<12.2 的旧环境，归约用 fp32 更安全；
    // 其他新环境可直接用 T（如 fp16）做归约，减少类型转换的代价。
    // 这是出于数值与平台兼容性的工程权衡。
#if (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR >= 2) || BNB_FP16_AVAILABLE
    using TReduction = T;
#else
    using TReduction = float;
#endif

    // 使用 CUB 的 block 级归约原语：所有线程在一个 block 内协作，做“最大值”归约。
    // TReduction 是参与归约的标量类型，THREADS 是 block 的线程数。:contentReference[oaicite:1]{index=1}
    using BlockReduceT = cub::BlockReduce<TReduction, THREADS>;

    // 【并行划分思想】
    //  - 采用“一行一个 block”的映射：blockIdx.x 对应 row_id。
    //  - 线程对该行做“条带式(strided)访问”：tid 访问 i = tid, tid+THREADS, tid+2*THREADS, ...
    //  - 先各自求“线程局部的 |x| 最大值”，再用 BlockReduce 求整行 absmax。
    //
    // 共享内存用于：
    //  - BlockReduce 的临时存储（CUB 需要）
    //  - 存放整行 absmax 以便后续所有线程复用（避免重复全局内存访问）
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ TReduction smem_row_absmax;

    const int row_id   = blockIdx.x;          // 当前处理的行
    const T*  row_data = A + (row_id * cols); // 指向这一行的起始地址（行主存）

    // 每个线程在自己的条带上找“局部绝对值最大”
    // 这里初始化为最小浮点（负无穷，或很小的负值）以便后面用 fmaxf 更新
    TReduction row_local_absmax = -FLT_MIN;

    // 条带式读取：i 从 tid 开始，每次步进 THREADS
    for (int i = threadIdx.x; i < cols; i += THREADS) {
        // 读取并取绝对值：
        //  - __ldcs() 是带缓存提示的加载 intrinsic（streaming cache hint），可影响缓存行为；
        //    这里配合 fabsf 计算绝对值（若 T 为 fp16，前面把 TReduction 设为适当类型以保持数值稳健）。:contentReference[oaicite:2]{index=2}
        const TReduction absval = fabsf(__ldcs(&(row_data[i])));

        // 若启用“稀疏分解”（SPARSE_DECOMP==1）：
        //   - 超过阈值的值视为离群值（outlier），不参与本行 absmax 的统计（只在量化阶段置 0）。
        //   - 这样可避免离群值把量化 scale 拉大，提升非离群值的量化分辨率。
        if constexpr (SPARSE_DECOMP) {
            row_local_absmax = fmaxf(row_local_absmax,
                                     (absval < TReduction(threshold)) ? absval : row_local_absmax);
        } else {
            // 常规：全量参与，按绝对值更新最大
            row_local_absmax = fmaxf(row_local_absmax, absval);
        }
    }

    // 用 CUB 在 block 内做一次“最大值”归约，得到整行的 absmax。
    // Reduce(..., cub::Max(), cols) 的“cols”参数用于告知参与范围/边界（实现可忽略超界）。
    const TReduction row_absmax = BlockReduceT(temp_storage).Reduce(row_local_absmax, cub::Max(), cols);

    if (threadIdx.x == 0) {
        // 仅由 0 号线程把结果写入：
        //   - 全局 rowStats[row_id]（便于核外或后续核使用）
        //   - 共享内存 smem_row_absmax（本行内统一使用的 scale 分母）
        rowStats[row_id]   = smem_row_absmax = row_absmax;
    }

    // 栅栏：确保全体线程在看到 smem_row_absmax 之前完成其写入。:contentReference[oaicite:3]{index=3}
    __syncthreads();

    // 【量化阶段（逐行）】
    // 线性标度：scale = 127 / absmax
    // - 127 是有符号 int8 的最大正值（-128~127），用它把 [-absmax, +absmax] 线性映射到 [-127, +127]。
    // - __fdividef 是单精度的快速除法 intrinsic。:contentReference[oaicite:4]{index=4}
    const float scale = __fdividef(127.0f, smem_row_absmax);

    // 仍然采用条带式并行写出量化结果
    for (int i = threadIdx.x; i < cols; i += THREADS) {
        float val = row_data[i];  // 读原值（此处以 float 计算更直观；若 T=__half，编译器会做相应转换）

        if constexpr (SPARSE_DECOMP) {
            // 稀疏分解：离群值( |val|>=threshold ) 不量化，直接写 0（保持“稀疏 + 分离 outlier”策略）
            // 非离群值：乘以 scale 后用“就近取整”为 int（__float2int_rn）再存为 int8。:contentReference[oaicite:5]{index=5}
            out[row_id * cols + i] = (fabsf(val) < threshold)
                                        ? __float2int_rn(val * scale)
                                        : 0;
        } else {
            // 常规：所有元素都按线性标度量化为 int8
            out[row_id * cols + i] = __float2int_rn(val * scale);  // 四舍六入五成偶（round-to-nearest-even）
        }
    }
}

```


### 小结
因此我们可以了解到大概的流程：
1.模型加载时，权重仍然是 FP16/FP32 格式
2.当模型被移动到 CUDA 设备时（通过 device_map="auto" 或手动 .to("cuda")），Int8Params 的 to() 方法会被触发 
3.INT8Params 会调用`bnb.functional.int8_vectorwise_quant()` 进行向量级量化。生成量化权重（CB）和缩放因子（SCB），将原始权重数据替换为量化后的 INT8 数据

## 8bit 线性层实现

### 运行时状态的准备 - Linear8bitLt

在完成 Int8Params 的准备后，我们还需要准备整个 8bit 线性层的实现。因为你不仅需要把参数量化成 8bit，还需要能保证上层调用的时候能够正常用你的 8bit 参数进行各种运算。要想真正跑起 `LLM.int8()` 的前向，它还需要一整套“运行时状态 + 流程控制”
```py
# 要阅读本段源码，你需要先：
# cd AIInfra/05Infer/06CompDistill/third-party/bitsandbytes/bitsandbytes/nn/modules.py
import torch
import torch.nn as nn
import bitsandbytes as bnb

# 继承 nn.Linear：保持与 PyTorch 线性层相同的用法/接口，
# 但在“上卡”(to("cuda"))时把权重量化为 INT8，并在 forward 里调用 bnb 的 INT8 matmul。
class Linear8bitLt(nn.Linear):
    """
    这个类是 LLM.int8() 算法在 PyTorch 里的“替代线性层”实现。
    正确用法：
      1) 先用 fp16/bf16 的权重初始化（可以直接从普通 nn.Linear 拷贝 state_dict）
      2) 调用 .to("cuda")（或 .cuda()）时触发量化，把 FP 权重转为 INT8 缓存格式
      3) forward 时走 bnb 的 INT8 矩阵乘，少量离群通道走 16-bit 路径并与 INT8 结果相加
    """

    def __init__(
        self,
        input_features: int,      # 输入特征数（与 nn.Linear 一致）
        output_features: int,     # 输出特征数
        bias=True,                # 是否使用偏置
        has_fp16_weights=True,    # 是否保留一份 FP16 权重（训练或特殊场景用）
        threshold=0.0,            # 离群值(outlier)阈值；>0 表示启用混合精度分解
        index=None,               # 可选：给外部索引/稀疏池用（实现细节）
        device=None,              # 初始设备
    ):
        """
        初始化流程：
          - 先按普通 nn.Linear 初始化出权重/偏置张量
          - 准备一个 bnb 的“乘法状态对象”(MatmulLtState)，把门限/是否保留 FP 权重等元数据放进去
          - 用 Int8Params 包装权重张量：这样在 .to("cuda") 时会自动执行量化，生成 CB/SCB
          - 注册一个 pre-hook：在 load_state_dict 前，有需要的话先重排权重布局
        """
        super().__init__(input_features, output_features, bias, device)

        # bnb 的“矩阵乘法状态”，保存运行时所需的元数据（阈值、是否有 fp16 权重、量化缓存等）
        self.state = bnb.MatmulLtState()
        self.index = index

        # 配置离群阈值/是否保留 FP16 权重
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights

        # 当设置了阈值且不保留 FP16 权重时，启用一个“池化(use_pool)”路径（实现细节：用于管理分解出的稀疏/离群通道）
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        # 用自定义的 Int8Params 包一层权重张量：
        #   - 在 .to("cuda") 时，Int8Params 会把 self.weight.data 从 FP16 量化成 INT8 “块”(CB) + 缩放因子(SCB)
        #   - 量化缓存(CB/SCB)会作为属性挂在 weight 对象上，forward 时由 bnb.matmul 使用
        self.weight = Int8Params(
            self.weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,  # 推理场景一般 False；训练/微调可 True
        )

        # 注册一个“加载前的预处理”钩子：
        #   - 某些检查点的权重布局需要在真正 load 之前重排（行/列主、转置等）
        #   - 具体逻辑由 maybe_rearrange_weight 实现（在 bnb 代码里）
        self._register_load_state_dict_pre_hook(maybe_rearrange_weight)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        覆盖保存逻辑：在常规 nn.Linear 的权重保存之外，
        额外把 SCB(缩放因子) 也写进 state_dict（若当前是已量化状态）。
        注意：
          - 量化后的 INT8 权重“CB”已经存在于 self.weight.data 里了，这里只需要补充保存 SCB
        """
        super()._save_to_state_dict(destination, prefix, keep_vars)

        scb_name = "SCB"  # 统一用这个键名保存缩放因子
        # 两种来源：
        #  1) 已经 .cuda() 量化过：SCB 在 self.weight.SCB
        #  2) 只做了 init_8bit_state：SCB 在 self.state.SCB
        param_from_weight = getattr(self.weight, scb_name)
        param_from_state = getattr(self.state, scb_name)

        key_name = prefix + f"{scb_name}"

        # 保留一个 weight_format 以向后兼容（新版本只用行主(row-major)）
        format_name = prefix + "weight_format"

        if not self.state.has_fp16_weights:
            # 只有纯 INT8 推理（不保留 FP16 副本）时才需要专门把 SCB 写入 checkpoint
            if param_from_weight is not None:
                destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)  # 0: row-major
            elif param_from_state is not None:
                destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
                destination[format_name] = torch.tensor(0, dtype=torch.uint8)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        加载逻辑：
          - 先让父类把常规权重/偏置加载完
          - 再额外处理 SCB（如果 checkpoint 里带了 SCB）
          - 注意：如果你要加载“已经量化的 checkpoint”，必须先 .cuda() 触发量化初始化，
                  否则这里没地方存 SCB（会报错提示先调用 module.cuda()）
        """
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        unexpected_copy = list(unexpected_keys)

        for key in unexpected_copy:
            input_name = key[len(prefix) :]
            if input_name == "SCB":
                # 如果还没 .cuda() 量化，weight.SCB 是 None —— 无处安放缩放因子
                if self.weight.SCB is None:
                    raise RuntimeError(
                        "Loading a quantized checkpoint into non-quantized Linear8bitLt is "
                        "not supported. Please call module.cuda() before module.load_state_dict()",
                    )

                # 拿到 checkpoint 里的 SCB，拷到权重对象上
                input_param = state_dict[key]
                self.weight.SCB.copy_(input_param)

                # 若 state 里也有 SCB，则保持与 weight.SCB 一致
                if self.state.SCB is not None:
                    self.state.SCB = self.weight.SCB

                # 既然已消费这个 key，就从“意外键”里移除
                unexpected_keys.remove(key)

    def init_8bit_state(self):
        """
        把量化缓存从 weight(CB/SCB)“转移”到 state 上：
          - forward 期间 bnb.matmul 从 state 里取这些元数据
          - 转移后把 weight.CB/SCB 清空，避免重复持有
        """
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        """
        前向计算：
          - 若是训练模式，记录 state.is_training 供内核/路径选择
          - 若 weight 里还留有 CB/SCB（比如刚刚 .cuda() 完），先搬到 state（init_8bit_state）
          - 把 bias 显式转换到与输入一致的 dtype（例如 x 是 fp16）
          - 调 bnb.matmul(x, weight, bias, state)：这一步会用 INT8 乘法 + 少量 16-bit outlier 路径
          - 若是纯 INT8 推理（不保留 FP16），且 state.CB 仍有缓存，回写到 weight.data（保持一致）
        """
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # PyTorch 不会自动把 bias 转 dtype，这里手动对齐到 x 的 dtype（例如 fp16）
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # 真正的 INT8 矩阵乘：bnb.matmul 会读取 state 里的 CB/SCB、threshold 等，执行 LLM.int8 路径
        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        # 纯 INT8 情况下，把最新的 CB 回写到 weight.data（保证权重张量与缓存一致）
        if not self.state.has_fp16_weights and self.state.CB is not None:
            self.weight.data = self.state.CB

        return out


```

最关键的计算是通过 bnb.matmul() 完成的：
```py
 out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
```
至于这段代码，其实就是在前向传播中，`bnb.matmul()` 会根据是否有离群值（outliers）选择不同的执行路径。

### 运行时的秘密 - Kernel 实现

这段代码的前向传播会通过一系列调用，其中我找到了比较关键的部分：
```py
# 这段的源码位于：
# cd AIInfra/05Infer/06CompDistill/third-party/bitsandbytes/bitsandbytes/autograd/_functions.py

  # Mixed Int8 Matmul + Dequant + Bias
output, subA = torch.ops.bitsandbytes.int8_mixed_scaled_mm(
    A,
    CA,
    state.CB,
    SCA,
    state.SCB,
    outlier_cols,
    bias,
)
   # Int8 Matmul + Dequant + Bias
output = torch.ops.bitsandbytes.int8_scaled_mm.default(
    CA, state.CB, SCA, state.SCB, bias=bias, dtype=A.dtype
)
```
显然，它为纯 INT8 和混合精度提供了两种 Kernel。那么为什么要提供两种呢，我不是已经量化成 INT8 了吗？这是因为 `LLM.int8()` 算法的一个核心设计理念：混合精度分解来处理离群值（outliers）。更具体的在`./05Quantization01.md`里面已经讲述的很清楚了。