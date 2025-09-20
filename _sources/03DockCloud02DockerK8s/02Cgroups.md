<!--Copyright © ZOMI 适用于[License](https://github.com/Infrasys-AI/AIInfra)版权许可-->

# 02. Cgroups：容器资源控制关键(DONE)

Author by: 张柯帆，ZOMI

> 注：以下讨论基于 Linux 操作系统环境

在上一篇关于容器隔离的文章中，我们详细介绍了实现“视图隔离”的核心技术——Namespace。其原理是通过修改进程的系统视图，让进程只能“看到”操作系统的部分资源，但这种视图层面的隔离终究是“障眼法”：对宿主机而言，这些“隔离进程”与其他进程并无本质区别，仍可能无限制抢占CPU、内存等物理资源。若某一容器耗尽宿主机资源，会直接影响其他容器甚至宿主机本身，因此必须通过技术手段限制容器的资源使用——Linux内核提供的**Cgroups**，正是解决这一问题的关键。

![虚拟机和容器的架构对比](../images/03DockCloud02DockerK8s/01Container01.png)

观察上图可发现：虚拟机方案通过**Hypervisor**组件模拟硬件，在虚拟化过程中间接实现了资源限制；而容器引擎（Container Engine）没有类似Hypervisor的硬件模拟能力，因此需依赖Cgroups完成资源管控。接下来，我们将从Cgroups的定义、版本、核心概念、使用方式等维度，全面解析其如何为容器提供资源隔离能力。

## 1. 什么是 Cgroups

Cgroups是Linux内核实现资源管控的核心机制，也是容器技术能够稳定运行的基础之一。本节将从定义、起源和核心功能三个维度，明确Cgroups的本质定位。

### 1.1 Cgroups 定义与起源

Cgroups（全称为Control Groups）是Linux内核提供的一种资源管理机制，其核心能力是：将系统中的进程及其子进程“分组”，按资源类别（如CPU、内存）为不同进程组设定管控规则，最终形成统一的资源管理框架。

从发展历程来看，2006年，Google工程师发起该项目，最初命名为“进程容器（process containers）”，目标是实现进程级的资源限制；2008年，该功能被合并至Linux 2.6.24内核，正式更名为Cgroups；如今，Cgroups已成为Docker、Kubernetes等容器技术，以及systemd等系统管理工具的底层依赖，是Linux生态中资源管控的“基础设施”。

简单来说，Cgroups的核心价值是：**为进程组提供“资源限制、统计、优先级控制”的标准化能力**，让上层应用（如容器）无需关心内核细节，即可实现精细化资源管控。

### 1.2 Cgroups 四大功能

Cgroups通过四大核心能力，覆盖资源管控的全场景需求，具体如下：

- **资源限制（Resource Limiting）**：为进程组设定资源使用上限。例如，限制某进程组的最大内存使用量，一旦超出上限，系统会触发OOM（Out of Memory）Killer机制，终止超限进程以保护其他资源；
- **优先级控制（Prioritization）**：为不同进程组分配资源优先级。例如，为核心业务进程组分配更高的CPU时间片权重，或提升其磁盘I/O优先级，确保关键服务在资源紧张时仍能正常运行；
- **审计与统计（Accounting）**：实时统计进程组的资源使用情况。例如，记录某进程组的CPU占用率、内存使用量、磁盘I/O次数等数据，为监控、计费等场景提供数据支撑；
- **进程控制（Control）**：对进程组执行统一操作。例如，批量挂起、恢复某进程组内的所有进程，或限制进程组内的进程创建数量，防止“fork炸弹”等恶意攻击。

## 2. Cgroups 两大版本

Cgroups随Linux内核迭代不断优化，目前主要存在v1和v2两个版本，两者在架构设计和功能支持上有显著差异。本节将对比两个版本的核心特点，以及当前的生态应用现状。

### 2.1 v1：多层级架构

Cgroups v1是首个稳定版本，目前仍被Docker等主流容器技术广泛使用，其核心设计是为每种资源创建独立层级（Hierarchy）：

- 层级与资源绑定：每种资源（如CPU、内存）对应一个独立的层级，例如CPU管控对应`/sys/fs/cgroup/cpu/`目录，内存管控对应`/sys/fs/cgroup/memory/`目录；
- 进程组独立配置：若需同时限制某进程组的CPU和内存，需在两个层级下分别创建同名进程组，并将进程加入两个层级的进程组中；

这种设计的优势是“资源管控灵活”，但存在明显缺陷：

- 层级管理混乱：多资源管控时需维护多个层级，易出现配置不一致问题；
- 资源协调困难：无法统一协调不同资源的限制逻辑（如内存超限与CPU调度的联动）。

### 2.2 v2：统一层级架构

为解决v1的缺陷，Cgroups v2进行了架构重构，核心设计是单一统一层级（Unified Hierarchy）：

- 所有资源共享一个层级：CPU、内存、磁盘I/O等资源的管控，均在`/sys/fs/cgroup/`目录下的统一层级中完成；
- 进程组自动继承资源规则：若将进程加入某进程组（如`/sys/fs/cgroup/my-container/`），该进程组关联的所有资源控制器（如CPU、内存）会自动对其生效；

此外，Cgroups v2还新增了“更可靠的进程追踪”“细粒度资源控制”等功能，例如支持基于“内存压力”动态调整进程优先级，或精确限制进程的“匿名页内存”使用量。

### 2.3 版本生态现状

尽管Cgroups v2在设计和功能上优于v1，但目前仍以v1为主流，原因如下：1）历史兼容性：大量现有工具（如Docker、Kubernetes旧版本）基于v1开发，迁移至v2需修改底层逻辑；2）生态成熟度：v1的文档、问题解决方案更丰富，开发者对其认知度更高。

不过，随着Linux内核版本升级（Linux 5.4+已默认支持v2），以及Kubernetes 1.25+对v2的全面支持，Cgroups v2正逐步成为主流，未来将全面替代v1。

## 3. Cgroups 核心概念

要理解Cgroups的工作原理，需先掌握“子系统、控制组、层级”三个核心概念——它们是Cgroups架构的基石，也是实现资源管控的逻辑基础。

### 3.1 三大核心概念解析

Cgroups的三个核心概念相互关联，共同构成资源管控的逻辑框架，具体定义如下：

- **子系统（Subsystem）**：又称“资源控制器”，是Cgroups管控某类资源的最小单元。例如，`cpu`子系统负责CPU资源管控，`memory`子系统负责内存资源管控，每个子系统对应一种特定资源；
- **控制组（Control Group）**：简称“Cgroup”，是进程的“分组单位”。一个控制组可包含多个进程，且可关联多个子系统（如同时关联`cpu`和`memory`子系统），进程加入控制组后，会自动遵循该组的资源规则；
- **层级（Hierarchy）**：是控制组的“组织形式”，以树状结构管理多个控制组。子控制组会继承父控制组的资源规则，例如父控制组限制内存使用1GB，子控制组的内存上限无法超过1GB。

三者的关系为层级是控制组的组织框架，子系统是控制组的资源管控规则，进程通过加入控制组，间接遵循子系统的资源限制。

### 3.2 子系统与配置参数

在容器场景中，常用的子系统包括`cpu`、`memory`、`blkio`、`pids`等，每个子系统通过特定配置参数实现资源管控。以下是各子系统的核心参数及使用场景：

1. `cpu`子系统

`cpu`子系统负责限制进程组的CPU使用，核心参数如下：

- `cpu.shares`：相对权重参数，默认值为1024。例如，进程组A的`cpu.shares`设为1024，进程组B设为512，在CPU资源紧张时，A获得的CPU时间是B的2倍（资源充足时该参数不生效）；
- `cpu.cfs_period_us`与`cpu.cfs_quota_us`：绝对限制参数，需配合使用：
  - `cpu.cfs_period_us`：CPU调度周期（单位：微秒），默认值为100000（即100毫秒）；
  - `cpu.cfs_quota_us`：周期内最大CPU使用时间（单位：微秒）；

  例如，将`cpu.cfs_period_us`设为100000、`cpu.cfs_quota_us`设为50000，意味着该进程组每100毫秒最多使用50毫秒CPU，相当于限制使用0.5个CPU核心。

2. `memory`子系统

`memory`子系统负责限制进程组的内存使用，核心参数如下：
- `memory.limit_in_bytes`：内存使用硬限制，即进程组的最大内存使用量。例如，设为`512M`表示该组进程总内存使用不能超过512MB，超出后系统触发OOM Killer；
- `memory.soft_limit_in_bytes`：内存使用软限制，仅在系统内存紧张时生效。例如，设为`256M`时，若系统内存充足，进程组可使用超过256MB的内存；若内存紧张，内核会回收超出部分的内存。

3. `blkio`子系统

`blkio`子系统负责限制进程组对块设备（如硬盘、SSD）的I/O访问，核心参数如下：

- `blkio.throttle.read_bps_device`/`blkio.throttle.write_bps_device`：限制块设备的读写速度（单位：字节/秒）。例如，`echo "8:0 104857600" > blkio.throttle.write_bps_device`表示限制对`8:0`（设备号）的写入速度为100MB/s；
- `blkio.throttle.read_iops_device`/`blkio.throttle.write_iops_device`：限制块设备的读写IOPS（单位：次/秒）。例如，`echo "8:0 1000" > blkio.throttle.read_iops_device`表示限制对`8:0`的读取IOPS为1000次/秒。

4. `pids`子系统

`pids`子系统负责限制进程组内的进程（含线程）创建数量，核心参数为`pids.max`：
- 例如，`echo "100" > pids.max`表示该进程组内最多可创建100个进程，超出后无法创建新进程，可有效防止“fork炸弹”等恶意攻击。

## 4. 如何使用 Cgroups

Cgroups通过“cgroupfs”特殊文件系统暴露给用户空间，用户可通过“读写文件”的方式配置Cgroups。本节将分别介绍“手动操作Cgroups”和“容器引擎中使用Cgroups”两种场景，帮助理解其实际应用方式。

### 4.1 手动操作 Cgroups

手动操作Cgroups的核心是“创建进程组→设置资源限制→加入进程”，以限制进程内存使用为例，具体步骤如下：

步骤1：进入 memory 子系统目录

Cgroups默认挂载于`/sys/fs/cgroup/`目录，memory子系统对应`/sys/fs/cgroup/memory/`，执行以下命令进入该目录：

```bash
cd /sys/fs/cgroup/memory
```

步骤2：创建进程组

通过`mkdir`命令创建名为`my-container`的进程组（系统会自动在该目录下生成memory子系统的所有配置文件）：

```bash
sudo mkdir my-container
```

步骤3：设置资源限制

通过`tee`命令将内存限制（512MB）写入`memory.limit_in_bytes`配置文件：

```bash
# 限制该进程组的最大内存使用为 512MB
echo 512M | sudo tee my-container/memory.limit_in_bytes
```

步骤4：将进程加入进程组

通过将进程PID写入`tasks`文件，将进程加入`my-container`进程组。例如，将当前Shell进程（PID通过`$$`获取）加入该组：

```bash
# 将当前Shell进程加入 my-container 进程组
echo $$ | sudo tee my-container/tasks
```

加入后，当前Shell及后续在该Shell中启动的进程，均会受到“512MB内存限制”的管控，若超出限制，系统会触发OOM Killer终止超限进程。

### 4.2 容器中 Cgroups 应用

在实际容器场景中，用户无需手动操作Cgroups——容器引擎（如Docker）已封装Cgroups的配置逻辑，只需通过命令行参数即可设置资源限制。

例如，通过`docker run`命令创建Nginx容器，并限制其CPU、内存、进程数量，具体命令如下：

```bash
docker run -d --name my-nginx \
  --cpus="0.5" \       # 限制CPU使用为 0.5 个核心（对应 cpu.cfs_period_us=100000、cpu.cfs_quota_us=50000）
  -m "512m" \          # 限制内存使用为 512MB（对应 memory.limit_in_bytes=536870912）
  --pids-limit 100 \   # 限制进程数量为 100（对应 pids.max=100）
  nginx
```

Docker的底层逻辑是：创建容器时，自动在Cgroups各子系统下创建同名进程组，将容器内的所有进程加入该组，并根据用户参数设置对应的Cgroups配置文件，最终实现资源限制。

## 5. Cgroups 与 Namespace 的协同工作

容器的“完整隔离”需依赖Namespace和Cgroups的协同——两者分别解决“视图隔离”和“资源隔离”问题，共同构建容器的独立运行环境。

### 5.1 解决不同维度隔离

- **Namespace**：解决“看得到”的问题，通过修改进程的系统视图，让容器内进程只能看到“专属资源”（如独立的PID、网络栈、文件系统），但不限制资源使用量；
- **Cgroups**：解决“用多少”的问题，通过为进程组设定资源上限，防止容器无限制抢占宿主机资源，确保多容器共存时的稳定性。

例如，某容器通过PID Namespace看到独立的PID树（容器内PID=1对应宿主机PID=1234），同时通过Cgroups限制其CPU使用为1核、内存为1GB——两者结合，既让容器“误以为独占系统”，又确保其不会影响其他容器。

### 5.2 容器隔离必要性

若只有Namespace无Cgroups：容器可无限制使用宿主机资源，某一容器耗尽CPU或内存后，会导致宿主机及其他容器崩溃；若只有Cgroups无Namespace：容器内进程可看到宿主机的所有资源（如其他进程、网络设备），易出现进程冲突（如PID重复）或安全风险（如访问宿主机敏感文件）。

因此，Namespace和Cgroups是容器隔离的“两大支柱”，只有两者协同，才能实现“安全、稳定、高效”的容器运行环境。

## 6. 总结与思考

Cgroups作为Linux内核的核心资源管控机制，通过“子系统、控制组、层级”三大概念，为容器提供了CPU、内存、磁盘I/O等资源的精细化管控能力。在实际应用中，Cgroups需与Namespace协同工作：前者限制资源使用量，后者构建独立视图，共同实现容器的完整隔离。

## 参考与引用

- https://labs.iximiuz.com/tutorials/controlling-process-resources-with-cgroups（Cgroups 实践教程）
- https://segmentfault.com/a/1190000045052990（Cgroups v1 与 v2 对比解析）
- https://www.kernel.org/doc/Documentation/cgroup-v2.txt（Linux Cgroups v2 官方文档）
- https://man7.org/linux/man-pages/man7/cgroups.7.html（Cgroups 核心概念官方手册）
