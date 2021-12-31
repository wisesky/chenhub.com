---
title: Python 下的多进程和多线程编程
toc: true
mathjax: true
top: false
cover: true
date: 2020-12-30 14:22:34
updated:
categories: Algorithms
tags:
	- Algorithms
	- OS
---

　　之前在并发写日志文件的时候简单的梳理了一下同步和异步的机制，不过对于Python的并发实践涉及还是比较浅显，现在这篇文章希望能彻底从代码角度来重新理解并发编程。

## 什么是进程(Process) 和 线程(Thread)

> 进程是操作系统分配资源的最小单元
>
> 线程是操作系统调度的最小单元
>
> 一个应用至少包括一个进程，而一个进程包含至少一个线程

　　通常来说，多进程和多线程都是实现多任务的方式，而多任务的设计无论是进程还是线程，都遵循Master-Worker 模式，Master负责分配任务，Worker负责执行任务。在Python编程模型下，这二者的实践方法也非常类似。

　　从效果上来看，进程间的独立性更高一点，这体现在一个进程崩溃，不会影响其他进程和主进程的执行，比如，Apache的多进程模式。另一方面，进程由于是OS直接创建和调度的，所以相同的代码可能在不同OS下效果会有差异，这体现在OS对同时并发的进程数目是有一个数目限制的，一般情况下，这是一个经验数目的限制，并不是你的系统无法承载这么多的进程数，虽然在大部分情况下，这个限制都是日常使用中无法达到的上界，但是我认为这就跟IPv4 和 32 位系统类似，只是在当时的条件下，定下的一个无法轻易达到的上限标准，在高速的发展迭代下，成为的日后亟待解决的遗留问题。其次，进程的创建开销也因OS不同也有所差异，比如Unix/Linux 的fork开销就远小于WIndows的进程创建。

　　多线程模式则是多进程的精细化调度，我的理解是，这源于超线程技术的发展，使得一个CPU内核通过常用的寄存器和Cache等CPU内部常用部件的堆叠，达到模拟出2个CPU内核的效果，这当然是一个很优秀的技术，因为确实从软件层面可以把一个CPU线程当作内核直接调用，同时性能上也获得了几乎线性的提升，甚至在Unix/Linux查看CPU参数的指令lscpu下，CPU(s)都是直接显示的内核的线程数目，而非内核数目，常常让初识Linux的我感到非常迷惑。我相信，基于超线程的性能提升，编译器和各种软件的迭代是不可能忽视这么重要且好用的技术的。不过对于一个没有经历过这个技术更迭的人来说，什么多线程，多进程，本质上不是一个东西吗？搞得神神叨叨的。除此之外，大部分进程和线程的介绍，都基于定义的扩展，认真来讲，光看定义，就完全看不出这二者的差异，何况扩展。所以关于线程，虽然我知道的很多，比如说，线程是进程的执行单元，一个线程崩溃，进程则直接挂掉，所有线程共享进程的内存等等，但是却有完全不明白为什么会有这些设置。现在看来，当初没有搞明白这些，着实是有点遗憾，除了教育方式和教材的原因，自己的求知欲也被这些陡峭的学习曲线给淹没了，因为在一个每一个术语都不大理解的地方，哪有好奇心和求知欲的容身之处，早就被从小被训练好的敬畏心给覆盖了，不懂，可是我又不敢问！扯远了，回到线程上，从线程的起源来理解，这一些的设置就豁然开朗了，线程作为执行单元，和CPU提供的线程接口强相关的，而CPU本身就是整个计算机系统的最主要的执行部件，理所当然的线程更接近执行向，而当线程执行失败的时候，整个进程的执行理所当然的受阻挂掉，线程的作为执行者，应当获得进程所有的资源，所以进程下的线程都具备进程的全部资源权限。

　　在OS层面，线程的实现是有区别的，Windows的多线程效率是比多进程要高的，所以微软的IIS服务器基于多线程，显而易见的稳定性问题不如Apache。不够随着逐步的发展，现在又都是多进程+多线程的混合模式，搞得确实头大。不过参照，RISC 和 CISC的发展，Intel 后来的x86 和x64虽然都仍然维持的了 CISC的接口，但是在微指令层面，仍然是RISC的技术，这也使得其仍然可以从RISC的发展中获益，想到，当初的计算机组成的教材，也是突兀的抛出了微指令的概念，最后学完微指令也摸不着头，直到看到计算机体系结构中关于RISC和CISC的部分，才从重新理解的微指令架构。这次层面上，进程和线程的在Apache和IIS的表现形式或许多少能得到一点启示。不过当今Nginx的事件驱动的异步IO设计大行其道，Python的协程多少还算是找对了方向。

　　一个很好的关于进程和线程的比喻解释，形象的道出了二者的关系

> - 计算机的核心是CPU，它承担了所有的计算任务。它就像一座工厂，时刻在运行。
> - 假定工厂的电力有限，一次只能供给一个车间使用。也就是说，一个车间开工的时候，其他车间都必须停工。背后的含义就是，单个CPU一次只能运行一个任务。编者注: 多核的CPU就像有了多个发电厂，使多工厂(多进程)实现可能。
> - 进程就好比工厂的车间，它代表CPU所能处理的单个任务。任一时刻，CPU总是运行一个进程，其他进程处于非运行状态。
> - 一个车间里，可以有很多工人。他们协同完成一个任务。
> - 线程就好比车间里的工人。一个进程可以包括多个线程。
> - 车间的空间是工人们共享的，比如许多房间是每个工人都可以进出的。这象征一个进程的内存空间是共享的，每个线程都可以使用这些共享内存。
> - 可是，每间房间的大小不同，有些房间最多只能容纳一个人，比如厕所。里面有人的时候，其他人就不能进去了。这代表一个线程使用某些共享内存时，其他线程必须等它结束，才能使用这一块内存。
> - 一个防止他人进入的简单方法，就是门口加一把锁。先到的人锁上门，后到的人看到上锁，就在门口排队，等锁打开再进去。这就叫"互斥锁"（Mutual exclusion，缩写 Mutex），防止多个线程同时读写某一块内存区域。
> - 还有些房间，可以同时容纳n个人，比如厨房。也就是说，如果人数大于n，多出来的人只能在外面等着。这好比某些内存区域，只能供给固定数目的线程使用。
> - 这时的解决方法，就是在门口挂n把钥匙。进去的人就取一把钥匙，出来时再把钥匙挂回原处。后到的人发现钥匙架空了，就知道必须在门口排队等着了。这种做法叫做"信号量"（Semaphore），用来保证多个线程不会互相冲突。
> - 不难看出，mutex是semaphore的一种特殊情况（n=1时）。也就是说，完全可以用后者替代前者。但是，因为mutex较为简单，且效率高，所以在必须保证资源独占的情况下，还是采用这种设计。

来自阮一峰的博客[进程和线程的简单解释](http://www.ruanyifeng.com/blog/2013/04/processes_and_threads.html)



## Python的多进程编程 multiprocessing

　　首先给出单进程顺序执行的测试代码，给出一个计算$8^{20}$的任务，同时辅以sleep 2s的任务。

```python
import time
import os

def long_time_task():
    print('当前进程: {}'.format(os.getpid()))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))

if __name__ == "__main__":
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    for i in range(2):
        long_time_task()

    end = time.time()
    print("用时{}秒".format((end-start)))
    
当前母进程: 33121
当前进程: 33121
结果: 1152921504606846976
当前进程: 33121
结果: 1152921504606846976
用时4.004350185394287秒
```

　　基本是sleep 的4s，计算任务基本不耗时间

　　接下来是多进程改写

```python
from multiprocessing import Process
import os
import time


def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))

if __name__=='__main__':
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p1 = Process(target=long_time_task, args=(1,))
    p2 = Process(target=long_time_task, args=(2,))
    print('等待所有子进程完成。')
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
    
当前母进程: 33121
等待所有子进程完成。
子进程: 34270 - 任务1
子进程: 34271 - 任务2
结果: 1152921504606846976
结果: 1152921504606846976
总共用时2.009559392929077秒
```

　　并行的效率得到体现，执行时间减半。

　　p.join()的理解是，主进程会等待子进程执行完毕，才开始继续从p.join()之后开始执行，否则主进程会直接输出总共用时，然后子进程接着执行完再输出。

```
当前母进程: 33121
等待所有子进程完成。
子进程: 34669 - 任务1
总共用时0.002809762954711914秒
子进程: 34670 - 任务2
结果: 1152921504606846976
结果: 1152921504606846976
```

　　最后就是对进程管理调度，由于OS的不同以及提高CPU利用率的需求，更是因为程序员懒得一个一个手动启动Process进程，产生的一个统一的进程管理的接口的需求，这就诞生的进程池Pool的接口。

　　Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果进程池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求。

下面介绍一下multiprocessing 模块下的Pool类的几个方法：

1. apply_async

函数原型：apply_async(func[, args=()[, kwds={}[, callback=None]]])

其作用是向进程池提交需要执行的函数及参数， 各个进程采用非阻塞（异步）的调用方式，即每个子进程只管运行自己的，不管其它进程是否已经完成。这是默认方式。

2. map()

函数原型：map(func, iterable[, chunksize=None])

Pool类中的map方法，与内置的map函数用法行为基本一致，它会使进程阻塞直到结果返回。 注意：虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程。

3. map_async()

函数原型：map_async(func, iterable[, chunksize[, callback]])
与map用法一致，但是它是非阻塞的。其有关事项见apply_async。

4. close()

关闭进程池（pool），使其不在接受新的任务。

5. terminate()

结束工作进程，不在处理未处理的任务。

6. join()

主进程阻塞等待子进程的退出， join方法要在close或terminate之后使用。



　　这里设计一个有意思设计用例，来验证并观察进程的行为模式是否按照理论上的理解运行，即是创建大小为4的Pool，却同时启动5个计算任务，最后观察运行时间，理论上应该是开始的4个进程是并行，后面的任务等前面的进程空出来之后，再开始安排进程来计算。

```python
from multiprocessing import Pool, cpu_count
import os
import time


def long_time_task(i):
    print('子进程: {} - 任务{}'.format(os.getpid(), i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))


if __name__=='__main__':
    print("CPU内核数:{}".format(cpu_count()))
    print('当前母进程: {}'.format(os.getpid()))
    start = time.time()
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
    
CPU内核数:64
当前母进程: 33121
等待所有子进程完成。
子进程: 37454 - 任务0
子进程: 37455 - 任务1
子进程: 37456 - 任务2
子进程: 37457 - 任务3
结果: 1152921504606846976
结果: 1152921504606846976
子进程: 37454 - 任务4
结果: 1152921504606846976
结果: 1152921504606846976
结果: 1152921504606846976
总共用时4.115360736846924秒
```

结果跟预想的基本一致，是并行的4个任务的2s+后续一个任务的2s，总共4s+

由于Python的GIL（全局解释器锁）的存在，多线程的代码实际上一个时刻只有一个线程在执行，所以如果要充分利用多核CPU资源，一般都是通过多进程来实现的。

　　Add，多进程的数据共享和通信，multiprocessing.Queue 使用实例

```python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    print('Process to write: {}'.format(os.getpid()))
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read:{}'.format(os.getpid()))
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
    
Process to write: 39190
Put A to queue...
Process to read:39191
Get A from queue.
Put B to queue...
Get B from queue.
Put C to queue...
Get C from queue.
```

##

## Python 的多线程编程 Threading 

　　接口其实跟多进程一样

```python
import threading
import time


def long_time_task(i):
    print('当前子线程: {} - 任务{}'.format(threading.current_thread().name, i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))


if __name__=='__main__':
    start = time.time()
    print('这是主线程：{}'.format(threading.current_thread().name))
    t1 = threading.Thread(target=long_time_task, args=(1,))
    t2 = threading.Thread(target=long_time_task, args=(2,))
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
    
这是主线程：MainThread
当前子线程: Thread-1316 - 任务1
当前子线程: Thread-1317 - 任务2
结果: 1152921504606846976
结果: 1152921504606846976
总共用时2.0029757022857666秒
```

连结果都跟多进程一样，不是说GIL导致始终是单线程的性能吗？观察任务的组成会发现，计算任务基本不耗时间，主要是sleep ,但是sleep在运行中是会被当作类似IO的操作，被识别为闲置的线程，这时候GIL直接就开始切换线程了，所以sleep的时间实际上是并行执行的，因为不依赖CPU计算，依此推测计算部分还是应该是单线程的，因为依赖CPU，那只需要增加计算任务的时间即可，那就加大指数为 $8^{1000000}$

```python
def long_time_task(i):
    print('当前子线程: {} - 任务{}'.format(threading.current_thread().name, i))
    time.sleep(2)
    t = time.time()
    print("结果: {}".format(8 ** 1000000))
    print('计算用时： ', time.time()-t)
# 单独执行时间 long_time_task(1)
计算时间 : 11.462863683700562
总共用时13.43696641921997秒
# 多线程时间 
计算时间 : 22.85654377937317
计算时间 : 23.088494300842285
总共用时25.10121512413025秒

```

结果上，总计算时间是翻倍的，但是每个线程的输出时间却是翻倍后的22s，而不是单线程的11s，考虑到并发执行，每个线程执行到任何一个时间，都有可能被中断，切换到其他进程，这里的单线程输出时间就可以理解了，那就是线程1执行玩计算任务，还没输出时间，就被切换到线程2的计算来了，最后统一输出线程1和线程2的时间，那么结果二者都是22s左右。总之，可以确定的计算密集型任务，Python的GIL充分保证的单线程的性能。

　　多线程的数据共享和通信，寻找线程安全的数据结构即可，queue.Queue就是的

```python
from queue import Queue
import random, threading, time


# 生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.queue = queue

    def run(self):
        for i in range(1, 5):
            print("{} is producing {} to the queue!".format(self.getName(), i))
            self.queue.put(i)
            time.sleep(random.randrange(10) / 5)
        print("%s finished!" % self.getName())


# 消费者类
class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.queue = queue

    def run(self):
        for i in range(1, 5):
            val = self.queue.get()
            print("{} is consuming {} in the queue.".format(self.getName(), val))
            time.sleep(random.randrange(10))
        print("%s finished!" % self.getName())


def main():
    queue = Queue()
    producer = Producer('Producer', queue)
    consumer = Consumer('Consumer', queue)

    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()
    print('All threads finished!')


if __name__ == '__main__':
    main()
```



## 计算密集型 vs IO密集型

　　对于计算密集型任务，python下的多线程改写是不会有性能提升的，只有多进程才能利用多核CPU，这是明显区别于其他语言的特点，因为在其他高效语言中，多进程和多线程都是作为并发的手段，同时改善程序性能的，而且本身Python作为脚本语言，在计算密集型任务面前运行效率就不高，此时更高阶的手段是用C语言重构 或者 引入高效的其他库来代替。

　　对于IO密集型，Python本身的多线程就能较好的应对，因为线程调度保证了IO任务的并行执行，此时就算是CPU有多个线程资源，任务本身就无法充分利用，所以Python的单线程限制并不构成瓶颈，就算用其他语言改写成真多线程，提升也不明显，所以类似web应用的IO密集型任务，Python足矣。

## 异步IO ，事件驱动模型， 协程，单进程单线程模型

　　考虑到CPU和IO之间巨大的速度差异，一个任务在执行的过程中大部分时间都在等待IO操作，单进程单线程模型会导致别的任务无法并行执行，因此，才需要多进程模型或者多线程模型来支持多任务并发执行。

　　现代操作系统对IO操作已经做了巨大的改进，最大的特点就是支持异步IO。如果充分利用操作系统提供的异步IO支持，就可以用单进程单线程模型来执行多任务，这种全新的模型称为事件驱动模型，Nginx就是支持异步IO的Web服务器，它在单核CPU上采用单进程模型就可以高效地支持多任务。在多核CPU上，可以运行多个进程（数量与CPU核心数相同），充分利用多核CPU。由于系统总的进程数量十分有限，因此操作系统调度非常高效。用异步IO编程模型来实现多任务是一个主要的趋势。

　　对应到Python语言，单线程的异步编程模型称为协程，有了协程的支持，就可以基于事件驱动编写高效的多任务程序。



## Conclude

　　用代码完整的梳理了一下并发下的进程和线程，同时总结了一下使用场景，很多知识都连接起来了，接下来一个重要的部分就是Python的核心部分，异步IO编程模型-协程

　　关于协程的实践部分，写篇文章会主要介绍。



References:

[一文看懂Python多进程与多线程编程](https://zhuanlan.zhihu.com/p/46368084)

[进程 vs 线程 廖雪峰](https://www.liaoxuefeng.com/wiki/1016959663602400/1017631469467456)

[线程间通信 Python3 CookBook](https://python3-cookbook.readthedocs.io/zh_CN/latest/c12/p03_communicating_between_threads.html)

