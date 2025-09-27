<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 负载均衡

author by: 张万豪

## 负载均衡

负载均衡（Load Balancing）是指**将网络流量、计算任务或工作负载等资源请求，智能地分配到多个后端服务器、计算资源或网络路径上**，**来提高系统的资源利用率**，其本质是解决资源分配的不均衡问题，从而提升整个系统的性能。

可以通过硬件或者软件来实现网络负载均衡

有专用的负载均衡器

![image.png](https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Load_Balancing_Cluster_%28NAT%29_zh_cn.svg/2560px-Load_Balancing_Cluster_%28NAT%29_zh_cn.svg.png)



软件则通过算法，实现，最常用的负载均衡算法是ECMP


## ECMP技术详解

AI集群中的网络流量模式宇传统数据中心有显著的区别。传统数据中心通常处理大量、短小、随机的“老鼠流”，例如网页浏览请求。而AI训练，尤其是集合通信操作（如 All-reduce、All-to-All），会产生数量较少但带宽极高、持续时间长的“大象流”。这些大象流具有高度的同步性和周期性，流量模式相对固定。另外，相较于传统数据中心，AI训练中会依赖计算节点之间的频繁数据交换，网络中的拥塞、丢包、延迟抖动都会导致GPU等计算单元的等待，所以对低延迟、高吞吐有更高的要求。



![image.png](https://media.licdn.com/dms/image/v2/D4D22AQGxHxeRF2MqcA/feedshare-shrink_800/feedshare-shrink_800/0/1697388393383?e=2147483647&v=beta&t=qrmAEMLCy6D2vQWS4nClriUVOWDl8NJQTgNcaqH4fb0   )

## AI集群中的负载均衡

ECMP在AI集群中的挑战：

- **哈希冲突与负载不均（Hash Polarization）**
- **静态与无感知性**
- **对集合通信的效率低下**

ECMP的优化方案：

1. **增强型ECMP (E-ECMP) 与队列对扩展 (QP Scaling)**
2. **动态与全局负载均衡方案**


## Spray and Reorder

当ECMP及其优化方案仍受限于“流”的粒度时，“Spray and Reorder”（喷洒与重排序）技术则将负载均衡的粒度推进到了更精细的“包”（Packet）级别，旨在实现理论上最完美的负载均衡

出现背景、技术详解



## 参考文献
[](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.linkedin.com%2Fposts%2Fziad-al-barqawi_elephant-flows-and-mice-flows-are-two-types-activity-7119362932466999298-3a3s&psig=AOvVaw2wUeNYBQGaFKRqytRcrPE9&ust=1758932673121000&source=images&cd=vfe&opi=89978449&ved=0CBgQjhxqFwoTCOjqwfuU9Y8DFQAAAAAdAAAAABAy)
[](https://translate.google.com/translate?u=https://zh.wikipedia.org/zh-cn/%25E8%25B4%259F%25E8%25BD%25BD%25E5%259D%2587%25E8%25A1%25A1&hl=en&sl=zh-CN&tl=en&client=search)