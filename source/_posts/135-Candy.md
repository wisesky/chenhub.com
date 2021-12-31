---
title: 135 Candy
toc: true
mathjax: true
top: false
cover: true
date: 2021-01-13 15:59:39
updated:
categories: Algorithms
tags:
	- Algorithms
	- LeetCode
---

> There are *N* children standing in a line. Each child is assigned a rating value.
>
> You are giving candies to these children subjected to the following requirements:
>
> - Each child must have at least one candy.
> - Children with a higher rating get more candies than their neighbors.
>
> What is the minimum candies you must give?
>
> **Example 1:**
>
> ```
> Input: [1,0,2]
> Output: 5
> Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
> ```
>
> **Example 2:**
>
> ```
> Input: [1,2,2]
> Output: 4
> Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
>              The third child gets 1 candy because it satisfies the above two conditions.
> ```

　　比较容易理解的题目，但是却是Hard难度。比较好入手，却是卡了好几天，最后依赖Discuss才勉强总结出结果，此为解法一；之后在其他 LeetCode的题解上发现了一个很简单的思路，却无法证明正确性，希望这里可以将其证明出来。

### 解法一

　　此题最先想到的思路就是，利用凹函数/凸函数的极小值和极大值的特性，来搜索上坡下坡，遇到上坡理所当然的将所分配的candy数目相较于左边的candy数目+1，如果是下坡，那么就当作上坡的逆序列，反向上坡，思路并没有问题，但是卡在一个非常奇怪的点，那就是峰值Peak 的选取，回头来看，只需要选择上坡和下坡之中的最大值即可，但是实现过程中，这个思路始终得不到重视，反而遇到了诸多乱七八糟的实现困难，最终卡了数天之后不得不放弃，这其中主要就是迟迟无法实现，导致对这个思路的自我怀疑，总是想投机取巧般的小改一下函数，来尝试新方法，结果有不断产生新问题，最后自己都有点迷失方向了。现在想来，似乎在无意义的实践上浪费了很多时间，相反真正应该关注的思路分析上，甚少得到进步。多少有点试图用战术上的勤奋来掩盖战略上的懒惰，情绪上，思路卡顿的时候，自己过分急躁，有点考场上，答不出来，弃之可惜的感觉，没找到正确的打开方式，却慌不择路，心态濒临崩溃，这种感觉无论是当时还在现在都是不愿想起的，看来我还是无法坦然面对自己，一点儿小挫折，就呼天喊地的，真是缺乏意志的考验。

　　回到上坡/下坡，其实关于上坡可以仅仅根据上坡的长度来计算分配的总candy数目，因为起点是1颗，接下来则是递增为1的等差数列，亦即是起点为1的连续递增数量，可以轻松得到candy_sum,下坡也是一样的，唯一不确定的就是peak_candy，在同时拥有上坡和下坡的长度的时候，也可以推断出是 max(left,right) + 1；最后需要解决的是相邻的rating相同的情况，2个相同的rating可以把序列分割成2个独立的序列，分别计算candy_sum ，最后合并；相同的相邻区间长度大于2的情况，除开首尾部分，中间的candy都可以被置为1。

　　归纳起来，可以把数组模式当作 bottom -> peak -> bottom -> equal 的组合，如果只出现bottom -> peak -> equal ，那么就当作前述模式的特例，这是一个比较有技巧性的归纳，因为针对此题，这个模式基本覆盖所有可能的上坡/下坡情况，特别是equal 可能出现在任何位置的时候，这个模式就被递归的细分下去了，这也是此解法正确性的核心。接下来只需要统计这个模式每一部分的长度即可，根据长度就能轻松计算出此序列的candy_sum

```python
def candy(self, ratings: List[int]) -> int:
    length = len(ratings)
    if length == 0:
        return 0

    start = 0
    sum_candy = 1 # init previous round bottom sum =1, because -1 each round to correct bottom double count problem
    i = 0
    while i < length-1: # cur = i, next = i+1
        # botton -> peak : peak not included
        while i<length-1 and ratings[i] < ratings[i+1]:
            i += 1
        left = i - start
        start = i
        # peak -> next bottom : peak not included
        while i<length-1 and ratings[i] > ratings[i+1]:
            i += 1
        right = i - start
        start = i
        # count peak_candy and sum_candy
        peak_candy = max(left, right) + 1
        sum_candy += (left+1)*left//2 + (right+1)*right//2 + peak_candy - 1 # - 1 because left bottom included by previous round
        # handle equal 
        while i<length-1 and ratings[i] == ratings[i+1]:
            i += 1
            sum_candy += 1
        start = i

    return sum_candy
```

　　需要注意的地方是，每次识别bottom -> peak -> bottom -> equal 模式的时候，第一个bottom 会被当作前一个模式的第二个bottom重复计算，所以每轮sum_candy最后都需要减去bottom的candy数目，也就是1，同时初始化的时候，前一轮的模式sum_candy初值为1，就是为了抵消前一轮重复计算的bottom，也就是把第一轮当作只包含一个值rating[0]来计算。



### 解法二

　　一个非常简单明了的算法是，所有小朋友初始的candy都是1，从左至右，只针对上坡的情况，递增candy；接着从右至左，把原本下坡的情况变成上坡，继续递增candy。这样就轻松的得到了正确的candy分配方案，由于初值都是1，使得相邻相同rating的小朋友candy一直是1。

　　这个算法正确性完全依赖于上坡，而针对此题，最确定的candy分配方案就是上坡，因为下坡可能回面临不知道应该candy应该减到多少的不确定性问题，全部变成上坡，那问题就迎刃而解了。

```python
def candy(self, ratings: List[int]) -> int:
    length = len(ratings)
    if length == 0:
        return 0

    candys = [1]*length
    pre = ratings[0]
    for i in range(1, length):
        cur = ratings[i]
        if cur > pre:
            candys[i] = candys[i-1]+1
        pre = cur

    post = ratings[-1]
    for j in range(length-2,-1,-1):
        cur = ratings[j]
        if cur > post and candys[j] <= candys[j+1]:
            candys[j] = candys[j+1] + 1
        post = cur
    return sum(candys)
```



### Conclusion

　　此题的关键是发现上坡的计算的确定性，遗憾的是，虽然我发现上坡时稳定正确的，但是却没有好好利用起来，反而将上坡作为一种情况单独讨论，遇到下坡和平坡的时候，单独讨论也还好，只是合并结果的时候遇到了很大的困扰，而且下坡和平坡的时候很多candy无法确定的问题没找到根本原因；没有发现在这多种情况中，上坡的唯一性，错失突破口。

　　此外，在遇到难关的时候，迟迟无法突破的局面，使得自己试图把注意力从思路解析转移到实践的方法，注定是自欺欺人，核心的地方，还是需要多一点耐心，急躁的心态往往是饮鸩止渴；另一方面，自己的这个多少有点急躁的性格，也要多加利用，比如说，预料得到自己在过久的无法打开局面下，极为容易滋生出放弃的习惯，尽量不要让自己在一个进度上卡的太久，积极持续采取措施让自己的进度运转起来，否则放弃的后果是自己难以接受的，也容易白费了之前的一片苦心。