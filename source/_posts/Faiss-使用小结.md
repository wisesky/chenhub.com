---
title: Faiss 使用小结
toc: true
mathjax: true
top: false
cover: true
date: 2021-12-31 11:24:48
updated:
categories: Machine Learning
tags:
  - Machine Learning
  - Deep Learning	
  - Neural Network

---

　　无论是项目中，还是实际的机器学习的研究探索中，经常会遇到的一个需求就是向量相似度的计算，尽管这是一个简单纯粹到仅仅只需要一行代码就可以解决的事情

```python
import numpy as np
d = 64
v1 = np.random.random((1,d))
v2 = np.random.random((1,d))
similarity = np.sqrt(np.sum(np.square((v1-v2))))
```

　　但是，当需要在亿级别的向量中查询最相似的 Top-k 个向量的时候，遍历所有向量，计算全部的相似度，最后排序尽管是理论可行的；但是程序运行时间是无法接受的，而且还会面临当向量维度比较大，比如 [10k, 10k]的np.array向量，就会占用巨大的内存空间，直接无法运行的情况；所以高效快速的查询向量相似度，是一个非常值得去探索和优化的问题。

　　也是在这个时候，发现了一个非常好用的高效的相似度和向量聚类库 Faiss : Facebook AI Similarity Search ，可搜索的向量数量级别完全取决于内存大小。实际体验效果极佳，API简单易用，速度非常快，只需要增添一点代码，就可以用GPU来加速运算，只不过在不知道其实现原理的情况下，对一些参数的设置会感到比较迷惑，特别是官方的文档写的有点跳跃，这也是最近需要攻克亿级向量的高效搜索的时候，专攻这些问题，以及依据实践经验总结的一点成果，顺带梳理一下思路，纠正以前对一些参数的错误理解。

## Faiss 实现加速的原理

　　在使用Faiss的时候首先需要基于原始的向量build一个索引文件，然后再对索引文件进行一个查询操作。在第一次build索引文件的时候，需要经过Train和Add两个过程；后续如果有新的向量需要被添加到索引文件，只需要一个Add操作从而实现增量build索引，但是如果增量的量级与原始索引差不多的话，整个向量空间就可能发生了一些变化，这个时候就需要重新build整个索引文件，也就是再用全部的向量来走一遍Train和Add，至于具体是怎么Train和Add的，就关系到Faiss的核心原理了。下面会详细分析。

![faiss base](faiss_base.jpg)

　　宏观来看，Faiss定义一套分析vectors分布的框架，通过train来实现，之后add的vectors都会依据train的分析结果，分门别类，有点类似于自适应的分桶结构，所以，最终Query阶段，也可以快速根据分桶的结果，定位到相关的向量簇，实现加速。具体的train算法有多种选择，在详尽考察了官方文档之后，意识到，最核心也是最常用的，也是Faiss文档Tutorial最开始介绍的那2种算法：

-  Product Quantizer : PQ
- Inverted File System: IVF

![faiss search](faiss_search.png)

### 精确搜索： Brute-Force Search

　　Baseline仍然是暴力搜索，基础原理基本就是遍历向量库，计算相似度，最后返回top-k的相似度的查询结果，不过可能是源于C++实现的原因，效率方面仍然远超自己的python实现，甚至是numpy的优化版本仍然望尘莫及。不得不说，Facebook在小工具的专注程度和极限探索方面，确实有一手。

```python
import numpy as np
import faiss

# data prepare
d = 64
nb = 100000 # database size
nq = 10000 # queries size
np.random.seed(32)
xb = np.random.random((nb, d)).astype('float32')
xb[ : ,0] += np.arange(nb) / 1000
xq = np.random.random((nq, d)).astype('float32')
xq[ : ,0] += np.arange(nq) / 1000

# CPU: basic usage
# index_factory: Flat
index = faiss.IndexFlatL2(d)
print(index.is_trained)
index.add(xb)
print(index.ntotal)
```

### PQ ：乘积量化

　　PQ的核心思路还是聚类，只不过是在维度切分之后的再进行聚类

![PQ_1](PQ_1.jpg)

#### PreTrain

　　 在做 PQ 之前，首先需要指定一个参数 M，这个 M 就是指定向量要被切分成多少段，在上图中 M=4，所以向量库的每一个向量就被切分成了 4 段，然后把所有向量的第一段取出来做 Clustering 得到 256 个簇心（256 是一个作者拍的经验值）；再把所有向量的第二段取出来做 Clustering 得到 256 个簇心，直至对所有向量的第 N 段做完 Clustering，从而最终得到了 256*M 个簇心。

![PQ_PT](PQ_PT.jpg)

　　做完 Cluster，就开始对所有向量做 Assign 操作。这里的 Assign 就是把原来的 D维的向量映射到 M 个数字，以 D=128，M=4 为例，首先把向量切成四段，然后对于每一段向量，都可以找到对应的最近的簇心 ID，4 段向量就对应了 4 个簇心 ID，一个 128 维的向量就变成了一个由 4 个 ID 组成的向量，这样就可以完成了 Assign 操作的过程 -- 现在，128 维向量变成了 4 维，每个位置都只能取 0~127，这就完成了向量的压缩。所以实际向量内存占用也会得到压缩。

　　实际上，就是把每个向量分M段，每段都各自聚类为NC个簇， 最后每个向量都可以表示成 M*NC 个簇心

#### Query

　　查询向量同样要计算其与 $4*256 $个簇心的距离，这样对于任意一个向量库的向量$[11,22,33,44]$，根据查询向量与簇心11,22,33,44的距离$d_{11},d_{22},d_{33},d_{44}$, 即可得到查询向量与向量库向量的距离$d=d_{11}+d_{22}+d_{33}+d_{44}$

![PQ_Query](PQ_Query.jpg)

　　上述示例这样通过PQ优化，只用 $4\*256$次$32(128/4)$维向量的距离计算，加上 $4\*N$次查表，即可得到查询向量Query与任意一个向量库向量vector 之间的距离，而如果是暴力计算，则有$N$次$128$维向量计算，后者的复杂度会随着$N$的增加而递增。

```python
# index_factory : PQ16x8 or PQ16
m = 16   # preferale: d=4*m ,number of subquantizers ,即是把维度d分成m段， code_size = d / m, 一般是 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64 and 96 bytes.
n_bits = 8       # bits allocated per subquantizer 默认为8
pq = faiss.IndexPQ (d, m, n_bits)        # Create the index
pq.train (xb)                       # Training
pq.add (xb)                          # Populate the index
D, I = pq.search (xq, k)     
```



### IVF ：聚类分桶

　　PQ是从向量维度方面寻找优化空间，但是在向量库中搜索最相似的Top-k个向量的时候，依然会遍历整个向量库，特别是向量库中的N特别大的时候，效率还有很大的提升空间。回顾以前了解过的案例，发现《数学之美》介绍用于在数以亿计的邮箱地址进行单个邮件地址匹配的布隆过滤器，还有在Word2Vec中，用于降低计算量的 Hierarchical Softmax的设计，比较接近这种优化需求；特别是布隆过滤器利用多个哈希计算来得到数个独立的Hash info 来唯一确定单个邮件地址，跟现在的LSH，Local Sensitive Hash设计思想相似，LSH更进一步通过线性映射的手段，保留了部分高维向量的距离信息，但是会有一定概率会丢失需要的距离信息，解决方法是采取多个独立的最好是正交的线性映射，这跟机械部件图纸的三视图原理接近，那么在多个独立的线性映射之后，仍然距离近的向量间，就可以分桶操作。所以IVF也是利用以距离信息为基础的分桶操作，这也是算法题中常用的$O(logn)$的手段，就是二分法更近一步的分桶法，也是Hierarchical Softmax，对单个查询向量，对比所有分桶的中心，快速锁定在某个子桶中，，随后让其在桶内进行更进一步的相似度遍历，第一步是模糊查询，第二步则是精确遍历。

　　IVF首先会依据指定的聚类中心nlist，将向量库自然聚类成nlist个桶，然后在查询阶段，得到查询向量与聚类中心的排名，在前nprobe最相似的桶内，执行精确查询。

```python
# index_factory: IVF100,Flat
nlist = 100 # ncentroids 分桶数目 通常是 4*sqrt(n) ～ 16*sqrt(n)，其中n为数据集大小
nprobe = 10 # 在多少个桶内精确search， 如果 nprobe == nlist ，那么等同于暴力搜索
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)
# 暴力完整搜索
D, I = index.search(xq, k)
# 指定精确搜索bucket的数目，加速搜索
index.nprobe = nprobe
D, I = index.search(xq, k)
```



### IVFPQ : IVF+PQ

　　Faiss最常用的接口，同时从维度和向量库聚类分桶的角度来优化相似度搜索和查询,减少需要计算的目标向量的个数，做法就是直接对库里所有向量做KMeans Clustering，假设簇心个数为1024。那么每来一个query向量，首先计算其与1024个粗聚类簇心的距离，然后选择距离最近的top N个簇，只计算查询向量与这几个簇底下的向量的距离，计算距离的方法就是前面说的PQ。

![IVFPQ](IVFPQ.png)

```python
# index_factory: IVF100,PQ8x8
nlist = 100  # ncentroids 分桶数目 通常是 4*sqrt(n) ～ 16*sqrt(n)，其中n为数据集大小
m = 8 ##每个向量分8段
n_bits = 8 # 每个向量编码大小 eg: 8(defatult), 12, 16
k = 4 ##求4-近邻
quantizer = faiss.IndexFlatL2(d)    # 内部的索引方式依然不变
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, n_bits) # 每个向量都被编码为8个字节大小
index.train(xb)
index.add(xb)
index.nprobe = 10                
D, I = index.search(xq, k)          # 检索
print(I[-5:])
```

### GPU

　　只需要简单的增加几行代码，就可以把上述的index都转换成在GPU上运行，尽管并非所有index都有GPU的实现，而且相同的index，可能也有些参数上的大小限制在CPU和GPU版本的实现上是有不同的，不过目前为止，除了GPU显存是使用瓶颈外，使用中暂未发现很大的不同。

```python
cpu_index = faiss.IndexFlatL2(d)
ngpus = faiss.get_num_gpus()
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.add(xb)
k = 4
D, I = gpu_index.search(xq, k)
```

### Index_factory

　　Faiss除了提供上述普通的Python API之外，还提供了一个更加简洁，同时可组合和自定义性比较高的构造Index的API:Index_factory，可以仅提供描述index的字符串，就可以快速构建想要的index，不过由于时间有限，没机会深入探索，而且文档中常常不加解释的给出这种快速构建的字符串，常常令我摸不着头脑。上述代码同时也给出了构建字符串，帮助快速理解其字符串意义。

## 加速效果对比

　　在8M的向量库中，分别进行1k,3k的查找，观察其加速效果,时间构成为 train_add + search  2个部分组成

|            | Flat        | IVF4096,FLAT (nprobe=100) | IVF4096,PQ64x8(nprobe=100) |
| :--------- | :---------- | :------------------------ | :------------------------- |
| 1k in CPU  | 41 +743 s   | 6294 + 288 s              | 6674 + 12 s                |
| 3k in CPU  | 41 + 1891 s | 6127 + 917 s              | 6675 + 31 s                |
| 1k in GPU  | OOM         | OOM                       | 446 + 0.18 s               |
| 10k in GPU | OOM         | OOM                       | 429 + 0.2 s                |

　　可以看出，Flat是不需要train的，只有add需要一定的时间，IVF在向量库train阶段，都需要较长的时间，但是一旦train完成，search时间会比Flat短很多，IVFPQ则在此基础上拥有更短的search时间，效果显著；search时间在1k 到3k的均是线性增长的。

　　GPU方面，由于8M的向量库太大，无法装入3090 24GB的显存，所以如果不利用PQ进行向量压缩，无法构建index，所以只有IVF + PQ才有结果，train时间也相对于CPU大幅缩短，search时间甚至可以忽略不计。

### GPU上的data batch

　　Pytorch上的NN：model + data都会占用大量的GPU显存，model 除了pipe line 流水线这种不常用的Trick，大部分时候都是常驻显存的，所以data一般都会分batch，只要 model + batch <= 显存 就可以防止显存爆满。Faiss的文档在某个隐蔽的角落里提到过需要自己实现data的batch运行，不过示例代码虽然利用 dataset 来实现分batch遍历，但是最终的gpu_index 仍然是把向量库全add进来了，只不过利用 把gpu_index -> cpu_index 来分组清空 gpu 显存，来分batch add，而且操控复杂，不容易复现。

　　好在，曾经接触过的其他的Facebook 的项目 LASER 中有相关使用Faiss的代码，发现同时对 query 和 database 进行 batch -> index -> search 的方案完全可行，而且还可以控制占用显存的大小，不过频繁的batch切换可能会费时间。

```python
# batched add and search 
def knnGPU(query, database, k, mem=5*1024*1024*1024):
    faiss.normalize_L2(query)
    faiss.normalize_L2(database)
    dim = query.shape[1]
    batch_size = mem // (dim*4)
    sim = np.zeros((query.shape[0], k), dtype=np.float32)
    ind = np.zeros((query.shape[0], k), dtype=np.int64)
    for xfrom in range(0, query.shape[0], batch_size):
        xto = min(xfrom + batch_size, query.shape[0])
        bsims, binds = [], []
        for yfrom in range(0, database.shape[0], batch_size):
            yto = min(yfrom + batch_size, database.shape[0])
            # print('{}-{}  ->  {}-{}'.format(xfrom, xto, yfrom, yto))
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(database[yfrom:yto])
            bsim, bind = idx.search(query[xfrom:xto], min(k, yto-yfrom))
            bsims.append(bsim)
            binds.append(bind + yfrom)
            del idx
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        for i in range(xfrom, xto):
            for j in range(k):
                sim[i, j] = bsims[i-xfrom, aux[i-xfrom, j]]
                ind[i, j] = binds[i-xfrom, aux[i-xfrom, j]]
    return sim, ind
```

　　实践证明，batch的切换时间并不构成瓶颈，相比于CPU的暴力搜索，甚至有了100倍的速度提升，这个结果是之前未曾预料到的。

|                   | Flat |
| ----------------- | ---- |
| 1k in GPU(batch)  | 14 s |
| 5k in GPU(batch)  | 16 s |
| 10k in GPU(batch) | 21 s |

 ## Conclude

　　最终，data batch的GPU方案完胜，暴力搜索的时间就完爆Faiss本身的各种优化算法，这还不算Faiss的优化算法，仍然可以在data batch 的GPU上叠加使用。



References:

[Faiss Github](https://github.com/facebookresearch/faiss)

[Facebook： 亿级向量相似度检索库Faiss 原理+应用](https://mp.weixin.qq.com/s/5KkDjCJ_AoC0w7yh2WcOpg)
