---
title: 139 Word Break
toc: true
mathjax: true
top: false
cover: true
date: 2021-01-13 18:04:29
updated:
categories: Algorithms
tags:
	- Algorithms
	- LeetCode
---

> Given a **non-empty** string *s* and a dictionary *wordDict*containing a list of **non-empty** words, determine if *s* can be segmented into a space-separated sequence of one or more dictionary words.
>
> **Note:**
>
> - The same word in the dictionary may be reused multiple times in the segmentation.
> - You may assume the dictionary does not contain duplicate words.
>
> **Example 1:**
>
> ```
> Input: s = "leetcode", wordDict = ["leet", "code"]
> Output: true
> Explanation: Return true because "leetcode" can be segmented as "leet code".
> ```
>
> **Example 2:**
>
> ```
> Input: s = "applepenapple", wordDict = ["apple", "pen"]
> Output: true
> Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
>              Note that you are allowed to reuse a dictionary word.
> ```
>
> **Example 3:**
>
> ```
> Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
> Output: false
> ```

　　Medium难度，入手时第一个想到的是DFS，之前Tree相关的题目做的多了，上手就忍不住用DFS练一下手，虽然正确性没问题，但是性能不行，最终卡在TLE始终无法优化。或许是被DFS卡的有点心烦，也可能是自己懒得从头开始分析此题的其他解法，也没有打算另辟蹊径去分析此题，最后发现简单的DP就可以解决，什么时候DFS好用，什么时候DP更有优势，自己还是没能找到这二者的辨别方法。

### 解法一：DP

　　能用DP的关键要素是最优子结构，但是自己常常跟DFS的递归搞混，不知道哪种递归是可以运用递归子结构，哪种无法用，所以此题虽然DFS解法比较直接，但是并没有立刻想到可以用DP来解的。实际上，此题是可以转化成递归子结构的，而且跟DFS递推式基本一致，只需要根据起点的wordBreak： True or False 以及 后续的substring是否在wordDict来判断即可：

$dp[i] = dp[i-j] \cap (s[j:i] \in wordDict), j \in [0:i]$

起点初始化为True，否则后面没有机会变成True

```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    length = len(s)
    if length == 0:
        return False
    set_words = set(wordDict)
    set_words.update('')
    dp = [True] + [False] * length # dp[i] mean s[ :i] can split in wordDict

    for i in range(length+1):
        for j in range(i):
            if dp[j] and s[j:i] in set_words:
                dp[i] = True
                break

    return dp[-1]
```



### 思路二： DFS （TLE）

　　首先想到的是DFS，递归的切分substring，如果substring在wordDict中，那么就继续递归下去，思路简单明了，但是隐隐感觉肯定还是有优化的地方，只是现在还没找到。

```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    if len(set(s)) > len(set(''.join(wordDict))):
        return False
    set_words = set(wordDict)
    r = self.helper(s, set_words)
    return r

def helper(self, s, set_words):
    if s in set_words:
        return True

    for i in range(len(s),0,-1):
        sub = s[ :i]
        if sub in set_words:
            if  self.helper(s[i: ], set_words):
                return True

    return False
```

### 

### Conclusion

　　Tree类的题目DFS做的有点多了，甚至都忘了，DFS，分治，贪心算法都是又可能转化为DP的，毕竟DP才是目前算法题的核心且比较有意思的部分。