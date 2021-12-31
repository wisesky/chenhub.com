---
title: GPU in Pytorch  并行和分布式实践
toc: true
mathjax: true
top: false
cover: true
date: 2020-11-25 15:07:07
updated:
categories: Pytorch
tags:
	- Pytorch
	- CUDA
	- GPU
---

　　虽然还没有机会用到CUDA集群，但是前段时间对协程和并行化的研究，让我忍不住想要探索一下如何在多个GPU下利用Pytorch加快训练的实践方法，算是为之后并行训练优化一个理论参考吧！

　　Pytorch 大体上有3种实现并行的接口（另外还有一种不利用接口的拼接模型的技巧，之后再单独讨论），分别是：torch.multiprocessing, nn.DataParallel, nn.parallel.DistributedDataParallel，如果是是GPU多卡运行，最佳实践是 nn.parallle.DistributedDataParallel,官方文档 [CUDA SEMANTIC](https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead) 是这么描述的：

> **Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel**
>
> Most use cases involving batched inputs and multiple GPUs should default to using [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) to utilize more than one GPU.
>
> There are significant caveats to using CUDA models with [`multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing); unless care is taken to meet the data handling requirements exactly, it is likely that your program will have incorrect or undefined behavior.
>
> It is recommended to use [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel), instead of [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) to do multi-GPU training, even if there is only a single node.
>
> The difference between [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) and [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) is: [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) uses multiprocessing where a process is created for each GPU, while [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) uses multithreading. By using multiprocessing, each GPU has its dedicated process, this avoids the performance overhead caused by GIL of Python interpreter.
>
> If you use [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel), you could use torch.distributed.launch utility to launch your program, see [Third-party backends](https://pytorch.org/docs/stable/distributed.html#distributed-launch).



### torch.multiprocessing

　　torch.multiprocessing 是 Python下 multiprocessing 的替代品，接口基本一致，并根据情况进行扩张，建议使用 python: multiprocessing.Queue 在进程中传递Pytorch对象。除此之外，还有很多坑，详见官方笔记 [MULTIPROCESSING BEST PRACTICES](https://pytorch.org/docs/stable/notes/multiprocessing.html), [并行处理最佳实践](https://pytorch.apachecn.org/docs/1.4/64.html)

```python
import torch.multiprocessing as mp
from model import MyModel

def train(model):
    # Construct data_loader, optimizer, etc.
    for data, labels in data_loader:
        optimizer.zero_grad()
        loss_fn(model(data), labels).backward()
        optimizer.step()  # This will update the shared parameters

if __name__ == '__main__':
    num_processes = 4
    model = MyModel()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
```



### DataParallel

　　DataParallel原理是，把model 副本copy到所有GPU上，其中每个GPU消耗数据的不同分区，相当于SIMD，把数据条目根据GPU数量重新分配

#### 单机DataParallel并行

```python
model = nn.DataParallel(model)
```

　　代码验证 outside model 数据维度  和 inside model 维度

```python
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
        

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
    
# 2 GPUs
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

优点是，通过消耗输入数据不同分区的方式加快训练过程，缺点是，如果单个模型太大，无法放入单个GPU，就无法运行，这时有个Trick可以把模型分段载入不同GPU，利用拼接的方式完成并行训练，该模型将单个模型拆分到不同的 GPU 上，而不是在每个 GPU 上复制整个模型(具体来说， 假设模型`m`包含 10 层：使用`DataParallel`时，每个 GPU 都具有这 10 层中每个层的副本，而当在两个 GPU 上并行使用模型时，每个 GPU 可以承载 5 层）

#### 单机模型拼接并行

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))
```

对于模型太大而无法放入单个 GPU 的情况，上述实现解决了该问题。 但是，分析原理的时候可能已经注意到，如果模型单个GPU可以载入，它将比在单个 GPU 上运行它要慢。 这是因为在任何时间点，两个 GPU 中只有一个在工作，而另一个在那儿什么也没做。 由于中间输出需要在`layer2`和`layer3`之间从`cuda:0`复制到`cuda:1`，因此性能进一步恶化。

除此之外，还有通过异步方式构建流水线的手段加速，这个Trick比较精巧和优雅，思想可以学习一下

#### 单机Pipeline 并行

```python
class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)

setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')
```

在2 GPUs上能有50%的提速，离100%还是有点差距，加速对比如下

###![1](1.png) 

### 

依次是 2-GPU并行， 1-GPU运行， 2-GPU Pipelining 并行 的运行时间对比



### nn.parallel.DistributedDataParallel

　　基于torch.distributed 的分布式数据并行，原理和DataParallel类似，但是是跨机器和跨设备级别的数据并行，在每台机器和每台设备上复制，并且每个这样的副本处理一部分输入， 在向后传递过程中，将平均每个节点的梯度，总之，是按批处理维度分块指定设备之间的输入来并行化给定模块的应用程序。

`DistributedDataParallel`可以通过以下两种方式使用：

#### 单进程多GPU

```python
torch.distributed.init_process_group(backend="nccl")
# device_ids will include all GPU devices by default
model = DistributedDataParallel(model) 

```

#### 多进程多GPU

　　强烈推荐的使用方式，在单机多GPU的情况下，单进程很容易由于GIL出现利用率不足的问题，这时候多进程就是唯一解决办法。最佳实践是，将DDP(DistributedDataParallel)配合多进程一起使用，每个GPU分配一个进程，会比torch.nn.DataParallel快得多，也是目前Pytorch最快的训练方法。

使用步骤：

1. 在N个GPU的单机上生成N个进程，这个过程可以交给torch.distributed.launch完成

```bash
python -m torch.distributed.launch --nproc_per_node=n distributed_data_parallel.py
```

2. 在代码中绑定GPU 编号，同时并行化model

```python
parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# 进程内绑定 GPU rank id
torch.cuda.set_device(args.local_rank)
# 构造model
torch.distributed.init_process_group(backend='nccl', world_size=n, init_method='env://')
model = torch.nn.parallel.DistributedDataParallel(
  									model,
										device_ids=[args.local_rank],
										output_device=args.local_rank)
```

Note:

1. nccl 后端是分布式训练使用的推荐的最快后端，适用于单节点和多节点分布式训练
2. nccl同时支持混合精度分布式训练
3. no_sync 用于禁用DDP进程之间的梯度同步，直到退出此上下文区域的第一个梯度Forward-Backward中进行梯度同步

```python
ddp = torch.nn.DistributedDataParallel(model, pg)
with ddp.no_sync():
for input in inputs:
	ddp(input).backward()  # no synchronization, accumulate grads
ddp(another_input).backward()  # synchronize grads
```



### apex.parallel.DistributedDataParallel

　　基本上是torch.nn.parallel.DistributedDataParallel的wrapper，同时调用的时候优化了NCCL的使用和简化了参数

``` bash
# 调用的时候，注意 n <= 每个节点的GPU数量 同时默认 1个GPU对应1进程
torch.distributed.launch --nproc_per_node=n distributed_data_parallel.py
# 会自动提供的参数目前已知的是:
# args.local_rank
# os.environ['WORLD_SIZE']

```

简化的要点：

1. model = DDP(model) 即可，无需再传递 devices_ids output_device

2. init_process_group 中的 init_method='env://'

   ``` python
   torch.distributed.init_process_group(backend='nccl',init_method='env://')
   ```

直接给出示例代码

```python
# distributed_data_parallel.py
import torch
import argparse
import os
from apex import amp
# FOR DISTRIBUTED: (can also use torch.nn.parallel.DistributedDataParallel instead)
from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

# Each process receives its own batch of "fake input data" and "fake target data."
# The "training loop" in each process just uses this fake batch over and over.
# https://github.com/NVIDIA/apex/tree/master/examples/imagenet provides a more realistic
# example of distributed data sampling for both training and validation.
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if args.distributed:
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
    model = DistributedDataParallel(model)
    # torch.nn.parallel.DistributedDataParallel is also fine, with some added args:
    # model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                   device_ids=[args.local_rank],
    #                                                   output_device=args.local_rank)

loss_fn = torch.nn.MSELoss()

for t in range(500):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

if args.local_rank == 0:
    print("final loss = ", loss)
```

更复杂的多精度调用见 [mixed precision training with DDP](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)



### DDP 保存和加载检查点

　　一般是使用torch.save 和 torch.load 来完成，但是在多进程下，优化方法是，仅在一个进程中保存，然后在其他所有进程中加载。

``` python
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```



### DDP 与模型拼接并行

　　DDP 还可以与多 GPU 模型一起使用，但是不支持进程内的复制。 您需要为每个模块副本创建一个进程，与每个进程的多个副本相比，通常可以提高性能。 当训练具有大量数据的大型模型时，DDP 包装多 GPU 模型特别有用。 使用此功能时，需要小心地实现多 GPU 模型，以避免使用硬编码的设备，因为会将不同的模型副本放置到不同的设备上。

``` python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

将多 GPU 模型传递给 DDP 时，不得设置`device_ids`和`output_device`。 输入和输出数据将通过应用程序或模型`forward()`方法放置在适当的设备中。

``` python
def demo_model_parallel(rank, world_size):
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

if __name__ == "__main__":
    run_demo(demo_basic, 2)
    run_demo(demo_checkpoint, 2)

    if torch.cuda.device_count() >= 8:
        run_demo(demo_model_parallel, 4)
```

注意上述setup 是通过 torch.multiprocessing.spawn 来完成多线程启动的，所以初始化方法也需要通过setup单独来指定

```python
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()
    
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```



### Conclusion

　　至此基本分析完所有Pytorch下的多GPU使用技巧，大体方案还是利用多进程的方式来规避GIL来提升性能，只需要用DDP 来包装model ，同时进行分布式对应的初始化，然后多进程启动即可加速训练，基本没有大的变动。



References:

[torch.nn](https://pytorch.apachecn.org/docs/1.4/75.html)

[DistributedDataParallel API](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html?highlight=distributeddataparallel#torch.nn.parallel.DistributedDataParallel)

[CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead)

[分布式数据并行入门](https://pytorch.apachecn.org/docs/1.4/34.html)

[用Pytorch编写分布式应用程序](https://pytorch.apachecn.org/docs/1.4/35.html)

[apex.parallel](https://nvidia.github.io/apex/parallel.html)

