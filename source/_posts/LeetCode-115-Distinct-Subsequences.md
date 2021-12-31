---
title: LeetCode 115. Distinct Subsequences
toc: true
mathjax: true
top: false
cover: true
date: 2020-12-09 17:52:16
updated:
categories: Algorithms
tags:
	- Algorithms
	- LeetCode
---

> Given two strings `s` and `t`, return *the number of distinct subsequences of `s` which equals `t`*.
>
> A string's **subsequence** is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., `"ACE"` is a subsequence of `"ABCDE"` while `"AEC"` is not).
>
> It's guaranteed the answer fits on a 32-bit signed integer.
>
>  
>
> **Example 1:**
>
> ```
> Input: s = "rabbbit", t = "rabbit"
> Output: 3
> Explanation:
> As shown below, there are 3 ways you can generate "rabbit" from S.
> rabbbit
> rabbbit
> rabbbit
> ```
>
> **Example 2:**
>
> ```
> Input: s = "babgbag", t = "bag"
> Output: 5
> Explanation:
> As shown below, there are 5 ways you can generate "bag" from S.
> babgbag
> babgbag
> babgbag
> babgbag
> babgbag
> ```
>
>  
>
> **Constraints:**
>
> - `0 <= s.length, t.length <= 1000`
> - `s` and `t` consist of English letters.



　　或许是由于之前做过这个题目的原因，所以面对这道Hard题目，心态比较冷静；再加上100题前后的几乎都是二叉树类别的题目，自己面对链表和树类别的题目基本没碰到困难，所以DFS用的手软，到这里有点停不下来的感觉。所以此时第一想法就想到用DFS，虽然预感对于字符串匹配类别的题目，DFS肯定会出现性能问题，但是作为验证算法思想的先导思路，还是打算先实现出来再看，毕竟用DFS写代码，代码真的很短，AC的时候更是有种 吊炸天的 爽快感；结果还是差一点AC；其次，可能是之前做过的原因，虽然第一时间没写出dp递推公式，但是写出了递推矩阵，演算了一番，没想到居然是对的，很容易的转换成了算法代码，结果干净利落的AC了，甚至比之前的AC的时候还要快一倍以上。虽然这不是自己第一次独立解决Hard题目，但是这题没有借鉴之前的代码，而且从构思算法道实践成功基本没遇到什么障碍，还是值得自己兴奋一阵子了。

### 解法一：DFS

　　此题如果要用递归的方法，那么就可以从最简单的字符串匹配入手，就是s和t都是单字符，直接就可以快速返回结果，这就可以精简成3种情况：

1. s == '' 即待匹配的模式t还没有消耗完，就直接没有s可供匹配了，返回0
2. t == '' 即待匹配的模式t消耗完，返回1
3. s==t 直接返回1
4. 递归处理

　　接下来，就开始要做递推式的处理，t的首字符t[0]如果在s中，那么就可以进行递归处理，考虑到s中可能有多个t[0]相同的字符，就要对s和t做不同的切分处理来交给递归处理，最后把所有返回的结果sum一下，就可以了

``` python
class Solution:
    # DFS
    def numDistinct(self, s: str, t: str) -> int:
        if len(t) == 0:
            return 1
        if len(s) == 0:
            return 0
        if s == t:
            return 1

        r = 0
        t0 = t[0]
        pos = -1
        news = s[pos+1: ]
        while t0 in news:
            pos = news.index(t0)
            news = news[pos+1: ]
            r += self.numDistinct(news, t[1: ])
        return r
```

测试用例通过率：106/116， 离AC只有一步之遥，目测发现是超长的字符串s导致递归性能急剧下降，那么可以确定的是算法思路是没有问题的。接着开始转入DP的算法构思。

### 解法二： 动态规划 Dynamic Programming (DP)

　　由于之前做过这个题，所以非常确定有一个DP算法存在，但是对于这个DP是怎么构建最优子结构的，完全想不起来了。所以，接下来就完全需要自己朝着DP思路构思DP矩阵了。

　　字符串匹配的DP算法有一个非常有名的最长公共子序列(LCS)问题，几乎所有讲解DP的教材都拿这个题目做过示例分析，当时自己对这个递推式还是很不理解，为什么$dp[i][j]$在$s[i]!=t[j]$的时候，可以由$dp[i-1][j]$和$dp[i][j-1]$构成？当初还是理解了好久，才想明白。或许是当初对这个分解的情况思索的很久的原因，所以但凡遇到DP算法的，都会潜意识的往LCS靠；而此题恰好也是字符串匹配的题目，思路方向似乎没什么太大问题。

|      |  r   |  a   |  b   |  b   |  b   |  i   |  t   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  r   |  1   |  0   |  0   |  0   |  0   |  0   |  0   |
|  a   |  0   |  1   |  0   |  0   |  0   |  0   |  0   |
|  b   |  0   |  0   |  1   |  1   |  1   |  0   |  0   |
|  b   |  0   |  0   |  0   |  1   |  2   |  0   |  0   |
|  i   |  0   |  0   |  0   |  0   |  0   |  3   |  0   |
|  t   |  0   |  0   |  0   |  0   |  0   |  0   |  3   |

对$dp[i][j]$的定义开始还是不太明确，最初以为是本层i和上层i-1的累加，最后发现不对，最后从意义上的分析，最终确定是来自上一层之前所有可能性的累加，因为$dp[i-1][ :j]$中累加表示上层匹配结束之后，所有可供下层提供入口的数量，而且这个入口只有在$s[i]==t[j]$的时候，才有效，否则直接置为0，最后的返回的结果应该是$sum(dp[-1])$。

最关键的步骤就是正确性验证，在提供的2个测试用例都无误之后，基本就可以开始步入写代码的过程了.

|      |  b   |  a   |  b   |  g   |  b   |  a   |  g   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  b   |  1   |  0   |  1   |  0   |  1   |  0   |  0   |
|  a   |  0   |  1   |  0   |  0   |  0   |  3   |  0   |
|  g   |  0   |  0   |  0   |  1   |  0   |  0   |  4   |

由于只需要上层的数据就可以计算本层， 但是并不是完全copy上层的数据，所以没办法用dp常用的边更新数据边覆盖的方式复用同一个List， 所以这里采用了2个List交替使用的方式来逐层更新

``` python
# dp
    def numDistinct(self, s, t):
        dp = [0]* len(s)
        for i,char in enumerate(s):
            if char == t[0]:
                dp[i] = 1

        for char in t[1: ]:
            count = 0
            dp_next = [0] * len(s)
            for i in range(len(s)):
                if s[i] == char:
                    dp_next[i] = count
                count += dp[i]
            dp = dp_next

        return sum(dp)
```

36ms AC， 比之前AC的DP还快上一倍，这是最初没想到的。



### Conclusion

　　这次顺利AC的主要原因还是思路方向的偶然正确性，这跟学生时代做数学题似乎有着异曲同工的地方，就是尝试的手段恰好是对的。如果尝试的手段恰好不对呢？无论是实践还是算法，我领悟到的结论是，多思考问题和事物的本质，才是快速接近正确答案有效的方式。上面的过程也可以发现，在有了正确的递归矩阵的引导下，代码的实现其实很轻松；反倒是，没抓住问题本质的随意和胡乱的尝试，才是导致心态急躁的根源，因为这种胡乱的尝试，似乎给了一种正在接近解决问题的错觉，虽然积累一点关于问题的经验，但是很多时候似乎只是积累无意义的失败经验而已；在适当的失败经验的基础上，更多的思考原因，在发现问题和原因的基础上的尝试，才是有价值和意义的，也才是最有效率的方式。

