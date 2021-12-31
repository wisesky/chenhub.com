---
title: LeetCode 33. Search in Rotated Sorted Array
toc: true
mathjax: true
top: false
cover: true
date: 2020-11-17 16:19:58
updated:
categories: Algorithms
tags:
	- Algorithms
	- LeetCode
---

### LeetCode 33. Search in Rotated Sorted Array

![](img1.png)

　　应该是二分法的变种优化，但是实际分析的过程中，自己的想法被太多的if条件给弄晕了，虽然勉强解出来了，但是更多是凭借临场发挥来作出的，至于宏观的算法原理，自己仍然理解不够透彻，借助讨论区的总结出几个好玩的解法

#### 解法一

　　最直接的想法是找到数组的偏移量，通过还原数组为排序数组的方式，最终利用二分法搜索target ，但是自己并没有想出可以$O(log(n))$搜索偏移量的方法，所以没有深入思考。借助讨论区的提醒，其实偏移量就是最小值的下标，所以只需要找到一种在$O(log(n))$下找到最小值的方法，这里仍然是用二分法，只不过start, mid, end 之间比较关系是不同的

- start 和 mid:

　mid > start : 最小值 in left

![](img3.jpg)

　mid < start : 最小值仍然 in left:

![](img2.jpg)

- mid 和 end

　　mid < end: 最小值 in left，end = mid

![](img6.jpg)

　　mid > end : 最小值 in right, start = mid

![](img7.jpg)

　　只需要mid 和 end就可以利用二分法搜索最小值，只不过需要注意的一点是，可能会出现死循环

![](img8.jpg)

　　主要原因应该是，当最小值in right 的时候，mid 原地更新，而在in right 的情况，实际上mid所在的value不可能是最小值，所以 start = mid + 1 即可

```python
while start < end:
	mid = (start + end ) //2
	if nums[start] > nums[end]:
		start = mid + 1
	else:
		end = mid
bias = start
```

　　最后利用偏移量，转换mid 为 偏移后的mid，二分搜索target

```python
start = 0
end = len(nums) - 1
while start <= end:
	mid = (start+end) // 2
	mid_pos = (mid + bias) 	% len(nums)
	val = nums[mid_pos]
	if target == value:
		return mid_pos
	if target < value:
		end = mid - 1
	else:
		start = mid + 1
    
return -1
```



Time Complexity: $O(log(n))$

Space Complexity: $O(1)$

#### 解法二

　　这里还是将被切分的2段分成2个有序数组来处理，比如,[4 5 6 7 1 2 3 ]分成 [4 5 6 7 ]和[1 2 3 ]，判断target位于其中的哪一段，然后将另一段变成-inf or inf，这样做的目的是可以用正常的二分法来搜索改变后的数组。

[ 4 5 6 7 1 2 3] ，如果 target = 5，那么数组可以看做 [ 4 5 6 7 inf inf inf ]。

[ 4 5 6 7 1 2 3] ，如果 target = 2，那么数组可以看做 [ -inf -inf - inf -inf 1 2 3]。

　　实践阶段，只需要判断nums[mid]的val，如果val跟target同一段，val不变，如果val跟target不通段，就要变成-inf or inf

- nums[mid] 和 target 同一段的条件: 

```python
nums[mid] > nums[0] and target > nums[0]
or
nums[mid] < nums[0] and target < nums[0]
```



- nums[mid] 和 target 不通段的条件：

```python
nums[mid] > nums[0] and target < nums[0]
or
nums[mid] < nums[0] and target > nums[0]
```

这样动态的变更nums[mid]的val，保持nums始终与target同段段那部分不变，不同段则变为-inf or inf

```python
lo = 0
hi = len(nums) - 1

while lo <= hi:
	mid = (lo+hi) // 2
	val = nums[mid]
	if (val > nums[0] ) == (target > nums[0]):
		pass
	else:
		val =  float('-inf') if target < nums[0] else float('inf')
   
	if val < target:
		lo = mid + 1
	elif val > target:
		hi = mid - 1
	else:
		return mid
      
  return -1
```

Time Complexity: $O(log(n))$

Space Complexity: $O(1)$

#### 解法三

　　基于一个事实，数组从任意位置劈开后，至少有一半是有序的

　　这里只需要判断二分法的其中一半是否有序， 再根据有序的这一部分判断target是否包含其中

```python
lo = 0
hi = len(nums) - 1

while lo <= hi:
	mid = (lo+hi) // 2
	val = nums[mid]
	if target == val:
		return mid
		# lo-mid 有序
	if nums[lo] <= val:
		if target >= nums[lo] and target < nums[mid]:
			hi = mid - 1
		else:
			lo = mid + 1
	else: # mid-hi 有序
		if target > nums[mid] and target <= nums[hi]:
			lo = mid + 1
		else:
			hi = mid - 1
```

Time Complexity: $O(log(n))$

Space Complexity: $O(1)$



### Conclusion

　　$log(n)$的时间复杂度，基本都是二分法的变体，平日自己对二分法过于忽视，导致理解不深刻，所以这个题目做起来才比较费劲。除此之外，解法二巧妙的利用了target 和 nums[mid]是否同一个分割排序段的条件，来简化判断，同时动态的更新nums[mid]值的方法很有启发性，比自己笨拙的设置一大堆if条件，结果自己都搞不清楚状况来的优雅许多。解法一二都是利用抽取出的关键信息来还原二分搜索法，解法三，则是试图找到target分段之前，获取排序信息，再以及排序信息搜索target，题目顿时变的简单了许多。



Reference:

[LeetCode 33. Search in Rotated Sorted Array](https://leetcode.wang/leetCode-33-Search-in-Rotated-Sorted-Array.html)

