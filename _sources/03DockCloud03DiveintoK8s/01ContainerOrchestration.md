<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 01. k8s 容器编排与作业管理
Author by: 何晨阳，ZOMI

k8s 本质上是面向数据中心的操作系统，容器管理和作业管理是云原生架构的核心支撑，理解这两个概念有助于帮助我们理解 k8s 如何管理超大规模系统的运行机制。

## 1. 容器编排

### 核心概念
- **Pod**：最小部署单元，包含 1 个或多个共享网络/存储的容器
- **Deployment**：管理无状态应用的 Pod 副本集（支持滚动更新/回滚）
- **Service**：提供负载均衡和服务发现（ClusterIP/NodePort/LoadBalancer）
- **Ingress**：管理外部 HTTP/HTTPS 流量路由

### 控制器模型
实现对不同的对象、资源的编排操作，核心就是通过控制器模型实现的。如下图所示，其首先会通过上半部分定义期望的状态，下半部分定义被控制对象的模版组成。

![CRI 架构](../images/03DockCloud03DiveintoK8s/01controller.png)

定义完期望状态后，会通过控制循环，来将目标对象调谐到指定的状态，执行逻辑如下：
```go
for {
  实际状态 := 获取集群中对象 X 的实际状态（Actual State）
  期望状态 := 获取集群中对象 X 的期望状态（Desired State）
  if 实际状态 == 期望状态{
    什么都不做
  } else {
    执行编排动作，将实际状态调整为期望状态
  }
}
```

### 关键功能
Kubernetes 通过声明式配置与控制循环机制，将容器编排抽象为四大核心能力：
- 资源分配智能化（调度/存储）
- 故障恢复自动化（自愈/滚动更新）
- 基础设施抽象化（逻辑资源池）
- 变更操作无损化（零停机更新）

主要功能如下所示：
| 功能 | 描述 | 配置示例 |
|------|------|----------|
| 自动调度 | 基于资源需求和节点亲和性 | `nodeSelector: {disk: ssd}` |
| 自愈能力 | 自动重启故障容器 | `restartPolicy: Always` |
| 滚动更新 | 零停机部署 | `strategy: {type: RollingUpdate}` |
| 存储编排 | 动态挂载持久卷 | `persistentVolumeClaim: my-pvc` |

### Deployment 配置示例

Deployment 是管理无状态应用的核心抽象对象，下面是一个配置示例，其定义了 Kubernetes Deployment 的 YAML 配置文件，用于部署一个 Nginx 服务的多副本实例：
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deploy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
```

## 2. 作业管理（Job Management）

Kubernetes 作业管理通过 Job 和 CronJob 实现了对非持续型任务的精细化控制，主要包括一次性任务和定时执行任务。

### 2.1 Job（一次性任务）
运行直到**成功完成**（退出码为 0）的离散任务，配置示例如下所示：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-processing
spec:
  completions: 6     # 需要完成的任务总数
  parallelism: 2     # 同时运行的 Pod 数量
  backoffLimit: 3    # 失败重试次数
  template:
    spec:
      containers:
      - name: processor
        image: data-tool:v3.2
        command: ["python", "/app/process.py"]
      restartPolicy: OnFailure  # 失败时自动重启
```

### 2.2 CronJob（定时执行任务）
下面定义了一个每天 3 点运行的任务示例：
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-report
spec:
  schedule: "0 3 * * *"  # 每天 3 点运行
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: report-generator
            image: report-tool:latest
          restartPolicy: OnFailure
```

### 关键参数

| 参数| 作用 | 示例值 |
|------|------|----------|
| backoffLimit | 失败重试次数 | 3 |
| activeDeadlineSeconds | 任务超时时间 | 3600 |
| successfulJobsHistoryLimit | 保留成功 Job 记录数 | 5 |
| failedJobsHistoryLimit | 保留失败 Job 记录数 | 2 |

### 3. 容器编排 vs 作业管理对比
下面比较了容器编排和作业管理各自适用的场景：
| 维度| 容器编排（Deployment） | 作业管理（Job/CronJob） |
|------|------|----------|
| 设计目标 | 长期运行服务 | 离散任务执行 |
| 生命周期 | 持续运行 | 运行到完成/超时 |
| 重启策略 | Always (默认) | OnFailure/Never |
| 扩缩容机制 | HPA 自动扩缩 | parallelism 手动控制并发 |
| 典型场景 | Web 服务/数据库 | 批处理/定时报表/数据迁移 |

### 4. 最佳实践

下面是使用时的推荐的一些最佳配置：

- **资源限制：为 Job 设置 resources.requests/limits 避免资源竞争。**
- **超时控制：使用 activeDeadlineSeconds 防止任务卡死。**
- **存储分离：Job 中挂载临时卷（emptyDir）避免数据残留。**
- **监控：通过 Prometheus 监控 Job 执行状态和时长。**

## 总结与思考
k8s 通过容器编排和作业管理，实现了大规模容器部署，解决了手动调度效率低、易出错等问题，通过 Job/CronJob 优化了批处理任务管理。借助这两大能力，实现了从基础设施到应用层的全栈自动化，构建了现代分布式系统的基石。

## 参考与引用
- https://www.cnblogs.com/BlueMountain-HaggenDazs/p/18147309（《深入剖析 Kubernetes》容器编排与 k8s 作业管理）
