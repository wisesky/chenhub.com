---
title: 146 LRU Cache
toc: true
mathjax: true
top: false
cover: true
date: 2021-01-13 18:08:17
updated:
categories:
tags:
---

> Design a data structure that follows the constraints of a **[Least Recently Used (LRU) cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU)**.
>
> Implement the `LRUCache` class:
>
> - `LRUCache(int capacity)` Initialize the LRU cache with **positive** size `capacity`.
> - `int get(int key)` Return the value of the `key` if the key exists, otherwise return `-1`.
> - `void put(int key, int value)` Update the value of the `key` if the `key` exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, **evict** the least recently used key.
>
> **Follow up:**
> Could you do `get` and `put` in `O(1)` time complexity?
>
>  
>
> **Example 1:**
>
> ```
> Input
> ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
> [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
> Output
> [null, null, null, 1, null, -1, null, -1, 3, 4]
> 
> Explanation
> LRUCache lRUCache = new LRUCache(2);
> lRUCache.put(1, 1); // cache is {1=1}
> lRUCache.put(2, 2); // cache is {1=1, 2=2}
> lRUCache.get(1);    // return 1
> lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
> lRUCache.get(2);    // returns -1 (not found)
> lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
> lRUCache.get(1);    // return -1 (not found)
> lRUCache.get(3);    // return 3
> lRUCache.get(4);    // return 4
> ```
>
>  
>
> **Constraints:**
>
> - `1 <= capacity <= 3000`
> - `0 <= key <= 3000`
> - `0 <= value <= 104`
> - At most `3 * 104` calls will be made to `get` and `put`.

　　Meidum难度，没想到的是用单链表实现起来，很多琐碎的细节需要填补，磕磕碰碰的AC了，查看Discuss发现直接就用双链表实现即可，这才发现题设并没有要求只用单链表来实现，此题用双向链表会省去之前的繁琐操作。

### 解法一： 单链表

　　在AC之前，用单链表实现会有一个比较繁琐的细节就是，获取node的pre索引用作删除node用，最开始懒得扩展结果，准备接口都混在 get/put 内部去实现，最后发现非常不利于debug；最后发现还是需要抽象出更基础的接口，pop,append,getPreNode，保证这几个接口的正确性，后面的get/put实现起来就非常轻松了，结果也证实了这一点，一次就AC

```python
class LRUCache:

    def __init__(self, capacity: int):
        self.key2prenode = {}
        self.node2key = {}
        self.root =  ListNode()
        self.tail = self.root
        self.size = 0
        self.capacity = capacity

    def pop(self):
        if self.size < 1:
            return
        cur = self.root.next
        post = cur.next
        self.root.next = post
        if cur in self.node2key:
            key = self.node2key[cur]
            self.key2prenode.pop(key)
            self.node2key.pop(cur)

        if post in self.node2key:
            post_key = self.node2key[post]
            self.key2prenode[post_key] = self.root

        return

    def append(self, key, value):
        node = ListNode(value)
        self.key2prenode[key] = self.tail
        self.node2key[node] = key
        self.tail.next = node
        self.tail = node
        return

    def getPreNode(self, key):
        if key in self.key2prenode:
            return self.key2prenode[key]
        return None
    
    def update(self, prenode):
        if prenode == None or prenode.next == None:
            return 
        cur = prenode.next
        post = cur.next
        if cur == self.tail:
            return

        tail_old = self.tail
        self.tail.next = cur
        cur.next = None
        self.tail = cur
        if cur in self.node2key:
            key = self.node2key[cur]
            self.key2prenode[key] = tail_old
        prenode.next = post
        if post in self.node2key:
            post_key = self.node2key[post]
            self.key2prenode[post_key] = prenode
        return

    def get(self, key: int) -> int:
        pre = self.getPreNode(key)
        if pre == None or pre.next == None:
            return -1
        else:
            val = pre.next.val
            self.update(pre)
            return val
        
    def put(self, key: int, value: int) -> None:
        pre = self.getPreNode(key)
        if pre == None or pre.next == None:
            if self.size == self.capacity:
                self.pop()
                self.append(key, value)
            else:
                self.append(key, value)
                self.size += 1
        else:
            pre.next.val = value
            self.update(pre)
            
        return
```

由于没想过用双向链表，在单链表的限制下，不得的面临获取pre node索引的繁琐问题，而且这个复杂的比想象中麻烦

### 解法二： 双向链表

　　如果是双向链表的话，问题就很简单了，而且可以在构建双向列表的元素直接从value变成key+value，简化之前的key2Prenode , node2key的复杂索引。由于自己并没有接触过Python下的双向链表，所以这里也没有打算自建双向链表，怕有坑，瞻前顾后不愿意尝试。实际上，双向链表的数据结构只是多了一个pre而已。

```python
class BiListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None

class LRUCache_Bi:
    def __init__(self, capacity) :
        self.capacity = capacity
        self.size = 0
        self.hash = dict()
        self.head = BiListNode()
        self.tail = BiListNode()
        self.head.pre, self.head.next = None, self.tail
        self.tail.pre, self.tail.next = self.head, None

    def get(self, key):
        value = -1
        if key in self.hash:
            node = self.hash[key]
            value = node.value
            self.remove_node(node)
            self.add_to_head(node)
        return value
    
    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self.hash:
            node = self.hash[key]
            self.remove_node(node)
            self.add_to_head(node)
            node.value = value
            return
        if self.size == self.capacity:
            node = self.tail.pre
            self.remove_node(node)

        newNode = BiListNode(key, value)
        self.add_to_head(newNode)
        return

    def add_to_head(self, node):
        post = self.head.next
        post.pre = node
        node.next = post
        self.head.next = node
        node.pre = self.head
        self.hash[node.key] = node
        self.size += 1
        return

    def remove_node(self, node):
        pre, post = node.pre, node.next
        pre.next, post.pre = post, pre
        self.hash.pop(node.key)
        self.size -= 1
        return
```

这里在get/put之外，重新抽象出add_to_head , remove_node 接口，对应单链表版本的append, remove ，然后在get/put直接调用即可，简单明了，这里需要注意的是，实现过程中，单链表的tail代表的是最新访问的node，而在这里双链表版本插好相反，head才是最新访问的node；此外，这里额外添加了一个hash字段，实际完成的就是key2node的功能。



### Conclusion

　　双向链表实际上是学生时代学的非常透彻的数据结构了，遗憾的是，当初没能从代码级别去理解，现在只是重新完成当初未完成的事情。奇怪的是，当初的二叉树和树结构理论基础也比较好，很顺畅的就可以切换到现在的代码；反而是更简单的链表，无论是理论还是实践都学的没有树结构好，可能是链表递归并不是常用方法，反而是树结构一般无脑递归就能解决绝大多数问题。最后，通过此题发现，在写数据结构和类的之前，一个好的抽象接口将会带来非常多的收益，这是自己一直不重视的，往往会看完题目就忙着开始动手写代码，这在往常的LeetCode中不是什么大问题，但是在这在类似架构搭建和设计的问题上，很容易自己给自己埋坑，以后自己在程序架构设计上，还需要多下功夫。

