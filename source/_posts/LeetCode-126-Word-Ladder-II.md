---
title: LeetCode 126. Word Ladder II
toc: true
mathjax: true
top: false
cover: true
date: 2020-12-15 11:33:20
updated:
categories: Algorithms
tags:
	- Algorithms
	- LeetCode
---

> Given two words (*beginWord* and *endWord*), and a dictionary's word list, find all shortest transformation sequence(s) from *beginWord* to *endWord*, such that:
>
> 1. Only one letter can be changed at a time
> 2. Each transformed word must exist in the word list. Note that *beginWord* is *not* a transformed word.
>
> **Note:**
>
> - Return an empty list if there is no such transformation sequence.
> - All words have the same length.
> - All words contain only lowercase alphabetic characters.
> - You may assume no duplicates in the word list.
> - You may assume *beginWord* and *endWord* are non-empty and are not the same.
>
> **Example 1:**
>
> ```
> Input:
> beginWord = "hit",
> endWord = "cog",
> wordList = ["hot","dot","dog","lot","log","cog"]
> 
> Output:
> [
>   ["hit","hot","dot","dog","cog"],
>   ["hit","hot","lot","log","cog"]
> ]
> ```
>
> **Example 2:**
>
> ```
> Input:
> beginWord = "hit"
> endWord = "cog"
> wordList = ["hot","dot","dog","lot","log"]
> 
> Output: []
> 
> Explanation: The endWord "cog" is not in wordList, therefore no possible transformation
> ```

　　这是一道似曾相识的Hard题，却意外的卡了好几天，好在最后还是凭借自己完成的算法方案，差一点AC，TLE的程度，最后试图优化无果，搜索Discuss，发现Python高赞方案跟自己的如出一辙，只不过在最费时的状态图构建上，做了很好的优化，借助这个优化思路，也很快获得了AC，学到了很好的优化经验。

　　说到似曾相识，是因为自己曾经做过的Frog Jump 也是一道Hard题，现在回想起来，那可能是自己独立实现AC的第一道Hard，这才有了后来渐渐想要不借助提示挑战Hard的勇气。虽说Frog Jump 是一道Hard，但是解题过程意外的轻松，很容易的就找到了算法方案，基本就是常规的思路，然后实现代码过程也没遇到很多问题，最后一次AC，倍感意外。彼时就有点反思道，是不是自己高估了Hard，所以被这个束缚，才一直觉得自己搞不定才没有去尝试。现在想来，可能只是因为经验不足，用于解题的算法工具不够充足才会有Hard无法突破的心理障碍，比如，Frog Jump , Jump Game 的多种变种以及这题Word Ladder 基本都跟算法（第四版）中的 确定有穷状态自动机DFA 和 不确定有穷状态自动机 NFA的解法一致，无论是最初AC，还是后来的Debug优化，基本都是围绕Pattern 构建的状态图流转即可。此题甚至都无需构建状态图，只需要寻找合适的数据结构来标记状态即可，多少跟图论算法比较相似。

### 算法思路分析

　　本质上，这其实是一道图论题目，在wordList提供的图之中，搜索从beginWord 到 endWord 的最短路径，最短路径可以直接用BFS求解，因为题设只需要输出最短路径的可能值即可，所以找到终点之后，程序即可中止。

　　算法上，还是延续之前的各种Jump的变种思路，难点应该还是在于实现上，毕竟这些变种题会有各种约束条件的变化导致代码上会有些许不同，这个恐怕要在写代码的时候才能意识得到。

### 算法实现

　　跟DFA/NFA类似，先把wordList转化成图表达，搜索所有的有连接关系的单词对，并保存起来。

```python
# 判断 字符串 s1 s2 是否有边连接
def isPair(self, s1, s2):
	c = 0
	for i,j in zip(s1, s2):
		if c > 1:
			return False
		if i != j:
			c += 1
	return c == 1
      
# word 在wordList 搜索所有可能的边
def constructidx(self, word, wordList, des2src):
	for w in wordList:
	if w != word and self.isPair(w, word):
		des2src[w].add(word)
		des2src[word].add(w)
	return
```

　　自己虽然知道在搜索连接边的isPair写的有点简单粗暴，有很大的优化空间，因为随着wordList的length增大，这个调用次数也是呈指数上升的，所以isPair的一点小优化能极大的降低TLE概率，事实也确实如此。无奈自己面对这样的基础优化，还是有点无能无力，似乎陷入了这个第一想法就得到结果的算法思路限制，无法彻底的想到新思路，最后在Discuss找到了这一段的简洁实现，利用wordList_set的在搜索阶段的改进很明显，除此之外以word为基础重新构建的方法会快一点，毕竟一个单词每一个位置都用26个字母重新替换一下，也需要$O(N)$的时间复杂度，综合起来$O(MN)$；而上述的代码可能需要$O(M^2)$， 其中M代表wordList 长度，N代表word本身的长度，这个角度来看，M较大的可能性还是比较高的。

```python
def constructidx_1(self, word, wordList_set, des2src):
	for i in range(len(word)):
		for char in string.ascii_lowercase:
			tmp = word[ :i] + char + word[i+1: ]
			if tmp in wordList_set:
				des2src[word].add(tmp)
				# des2src[tmp].add(word)
	return
```

　　也是这一段的优化最终将TLE的代码转化为AC的代码

　　接下来就是BFS搜索的过程了，自己在写这段代码的时候，不知道是不是脑子抽风了，又犯了还没想好整体构思，就提枪上阵的毛病，本来是一段很基础的BFS搜索的过程，结果由于访问状态标记的代码没想明白，就随意的放置访问状态的代码，结果调试了1天，才发现问题所在，期间出现了各种匪夷所思的访问状态变化的问题，搞得自己经常莫名其妙，各种怀疑是不是其他地方出了问题。实在是有点不应该。

　　可能是以前的BFS都需要遍历所有的节点之后，自动返回，所以直接采用queue的方式，queue为空自动返回即可，不需要标记BFS目前在第几层，然而此题是需要求最短路径，显然是需要采取层次遍历的方式依此进行，所以用q, q_next依此交替的方式，q是目前的遍历层次，q_next是接下来需要遍历的层次；其次，状态标记visited 的问题，此题由于要输出所有可能的最短路径，所以不仅进入q_next的新节点需要标记回溯路径，有可能q_next中的某个节点a，在此层被2个不同的节点b和c连接，此时需要把这2条路径需要标记回溯路径，之前这种可能性被自己忽略了，所以每当有多条的路径的时候，自己的代码只输出了一条，最后发现是遍历b和c的时候，b-a 使a 进入q_next, 并且visited[a] = True,标记b-a的回溯路径， 而当 c-a 需要标记回溯路径的时候，却由于visited[a]==True 被跳过了，所以才有那段不伦不类的if elif 的分支判断

```python
q = deque()
q.append(endWord)
visited = {endWord:True}
bp = defaultdict(set)
#res = defaultdict(list)
#res[endWord] = [ [endWord] ]
while len(q) > 0 :
	q_next = deque()
	for e in q:
		# self.constructidx(e, wordList_set, des2src)
		self.constructidx_1(e, wordList_set, des2src)
		for src in des2src[e]:
			if not visited.get(src, False)  :
				visited[src] = True
				q_next.append(src)

#				r = [ [src] ]if len(res[e]) == 0 else [[src] + re for re in res[e]]
#				res[src].extend(r)
				bp[src].add(e)
			elif src in q_next:
#				r = [ [src] ]if len(res[e]) == 0 else [[src] + re for re in res[e]]
#				res[src].extend(r)
				bp[src].add(e)
	if beginWord in q_next:
		break
	wordList_set = wordList_set - set(bp.keys())
	q = q_next
  
# return res[beginWord]
res = []
self.backPath(beginWord, endWord, [], res, bp)
return res

# 回溯还原最短路径
def backPath(self, start,endWord, r ,res,bp):
        if start == endWord:
            res.append(r+[endWord])
            return

        for st in bp[start]:
            self.backPath(st, endWord, r+[start], res, bp)

        return
```

上述注释的代码，是除了回溯之外，另一种BFS中，直接保存结果的方法，当初以为是TLE的主因，最后证明其实不是，所以也是一种输出结果的方法，原理是用字典res，保存所有以word开头的可能的路径列表。

### BFS优化

　　Discuss Python高赞解法大体BFS跟我的一致，但是我前面也分析过，我这个奇怪的if/elif分支其实还是有很大的优化空间，再则自己的状态划分不佳其实也是导致这么奇怪分支的一个原因，所以高赞的解法直接把这二者完美的统一了，窃以为也是一个简洁优雅的解法，值得学习。这里res 跟我上述的res相反，是保存以word结尾的路径列表的字典

```python
q = defaultdict(list)
q[beginWord] = [[beginWord]]
wordList_set = set(wordList)
res = []
while len(q) > 0:
  q_next = defaultdict(list)
  for word in q:
    if word == endWord:
      res.extend(k for k in q[word])
    for i in len(word):
      for char in string.ascii_lowercase:
        tmp = word[ :i] + char + word[i+1: ]
        # 未访问状态的列表中搜索拼接的可能的值，逐渐减少搜索空间，
        #也是一种很好的优化方法，即逻辑上实现了 visited 状态验证，
        #同时一层一更新减少了搜索空间，用set实现极大的优化了搜索时间
        if tmp in wordList_set:
					q_next[tmp] = [r + [tmp] for r in q[word]]
          
	# wordList_set 实际上是保存了所有未访问状态的列表，
  #BFS的时候每一层更新一次，避免奇怪的if/elif 分支判断，
  #是更准确的状态划分方法
	wordList_set = wordList_set - set(q_next.keys())
  q = q_next
return res
```

### Conclusion

　　算法优化阶段的时间复杂度分析做的不是很到位，其实似乎还是懒居多，细想一下，M和N之间的对比，似乎也能找到一点蛛丝马迹；其次，BFS状态划分没有因地制宜，还是想以套路直接鲁莽上阵，其实在发现应该按照层次BFS的时候，此题应该已经解决了，解析之后的后续优化工作还是需要多积累一些套路经验，特别是Python本来就比较慢的基础上，时间复杂度分析的重要性可能会越来越大。

