---
title: 147/148 merge sort
toc: true
mathjax: true
top: false
cover: true
date: 2021-01-21 15:24:50
updated:
categories: Algorithms
tags:
	- Algorithms
	- LeetCode
---

> Given the `head` of a linked list, return *the list after sorting it in **ascending order***.
>
> **Follow up:** Can you sort the linked list in `O(n logn)` time and `O(1)` memory (i.e. constant space)?
>
>  
>
> **Example 1:**
>
> ![img](https://assets.leetcode.com/uploads/2020/09/14/sort_list_1.jpg)
>
> ```
> Input: head = [4,2,1,3]
> Output: [1,2,3,4]
> ```
>
> **Example 2:**
>
> ![img](https://assets.leetcode.com/uploads/2020/09/14/sort_list_2.jpg)
>
> ```
> Input: head = [-1,5,3,4,0]
> Output: [-1,0,3,4,5]
> ```
>
> **Example 3:**
>
> ```
> Input: head = []
> Output: []
> ```
>
>  
>
> **Constraints:**
>
> - The number of nodes in the list is in the range `[0, 5 * 104]`.
> - `-105 <= Node.val <= 105`

　　排序类别的题目应该是所有排序算法的基础了，不过用单链表形式出现的还比较新颖，147是插入排序则是比较熟悉的，很轻松的就AC了，然后148则遇到了一点困难，扫完题目就发现当初比较偏爱的三向快速排序法完美适配，论速度应该没有比快排更快的了，但是意外没能通过OJ，可能是跟单链表形式的数组跟快排本身还是存在适配问题，如果是堆排序，明显树结构更合理一点，剩下的就只有归并排序了，奈何，自己从一开始就对归并排序不是特别来电，一直以来并没有太深入研究其算法原理，所以这里实现起来遇到了一点困难，更不知道除了自顶向下的归并排序之外，还有一个自底向上的归并排序，这样不仅算法复杂度满足O(nlogn)，空间复杂度也可以做到O(1)

### 解法一：自顶向下归并排序

　　虽然归并基本是以分治为主，递归的切分数组，最后合并2个已经排好序的数组，所以只需要处理初始化的状态就可以了，但是不是很确定这个初始化应该放在归并中还是放到切分的函数中，理论上应该是在切分函数里，因为主要的递归调用是调用的切分函数的，但是自己无时不刻总是担心的性能问题作祟，在这个问题上犹豫不决，更加犹豫的是，心理上有点怀疑这么粗暴的递归，真的能达到O(nlogn)的性能吗？结果证明是自己多虑了，真实的递归就是如此，枉费当初还从算法复杂的上推理过分治的原理公式，这么快就忘记了。

　　单链表下的分支惟一需要处理的问题就是切分了，需要手动切段单链表，同时处理递归的初始化状态，接下来专心写归并的递归即可，如果合并2个已经排序的单链表，这个实现起来还是比较简单的。

```python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        mid, tail = head, head.next
        while tail and tail.next:
            tail = tail.next.next
            mid = mid.next
        start = mid.next
        mid.next = None
        left, right = self.sortList(head), self.sortList(start)
        return self.mergeSort(left, right)

    
    def mergeSort(self, l1, l2):
        root = ListNode()
        tail = root
        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 or l2
        return root.next
```

### 解法二：自底向上归并排序

　　主要原理基本就是把递归的代码改写成迭代的代码，方法跟希尔排序类似，递增已经排序的size大小，初始size=1，也即是单个元素默认是已排序状态，随后依次合并2个已排序链表，这里主要的改写部分都是和单链表的切分有关，因为跟数组列表可以在O(1)时间内获取任意索引的元素不同，单链表有些繁琐的细节处理，这里的技巧是，根据size大小，获取需要合并的2个链表的头，然后在merge阶段，返回合并后的链表的tail，因为这个tail在下一轮的merge的时候需要用到，因为需要把上一个合并后的tail.next = 合并后的LinkList的head。

```python
class Solution:
    # buttom up merget
    def getSize(self, head):
        count = 0
        while head:
            count += 1
            head = head.next
        return count

    def splitLink(self, head, size):
        cur = head
        for _ in range(size):
            if not cur:
                break
            cur = cur.next
        if not cur:
            return None
        next_start = cur.next
        cur.next = None
        return next_start

    def mergeSort_bottomup(self, l1, l2, pre_tail):
        cur = pre_tail
        while l1 and l2:
            if l1.val<l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        while cur.next:
            cur = cur.next
        return cur

    # bottom up sort
    def sortList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head

        length = self.getSize(head)
        root = ListNode()
        root.next = head
        # pre_tail = None
        start = None
        size = 1
        while size < length:
            pre_tail = root
            start = pre_tail.next
            while start:
                left = start
                right = self.splitLink(left, size)
                start = self.splitLink(right, size)
                pre_tail = self.mergeSort_bottomup(left, right,pre_tail)
            size *= 2
        return root.next
```



### Conclusion

　　关于归并的性能还是不太熟悉，所以针对这么精准需要用到归并的时候，遇到了一点困难，不过好在大体的原理还是知道的，只是实现上卡在一些小地方；此外我还是始终认为快排才是最快的，只是无法满足O(1)的空间复杂度，所以在个别测试用例上，很容易TLE，而且快排的代码真的简洁明了很多，相对于归并来说。