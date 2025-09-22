import React, { useState, useEffect } from 'react';

const App = () => {
  const [activeCategory, setActiveCategory] = useState('Arrays');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedProblem, setSelectedProblem] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // DSA Problems Data Structure
  const dsaProblems = {
    'Arrays': [
      {
        id: 1,
        title: 'Two Sum',
        description: 'Find two numbers in the array that add up to a specific target.',
        solution: `def twoSum(nums, target):
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    return []`
      },
      {
        id: 121,
        title: 'Best Time to Buy and Sell Stock',
        description: 'Maximize profit by choosing one day to buy and another to sell.',
        solution: `def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit`
      },
      {
        id: 88,
        title: 'Merge Sorted Array',
        description: 'Merge two sorted arrays into one sorted array.',
        solution: `def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1`
      },
      {
        id: 217,
        title: 'Contains Duplicate',
        description: 'Check if an array contains duplicates.',
        solution: `def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False`
      },
      {
        id: 238,
        title: 'Product of Array Except Self',
        description: 'Return an array where each element is the product of all other elements.',
        solution: `def productExceptSelf(nums):
    n = len(nums)
    res = [1] * n
    left = right = 1
    for i in range(n):
        res[i] *= left
        left *= nums[i]
        res[n - 1 - i] *= right
        right *= nums[n - 1 - i]
    return res`
      },
      {
        id: 53,
        title: 'Maximum Subarray',
        description: 'Find the contiguous subarray with the largest sum.',
        solution: `def maxSubArray(nums):
    max_sum = current_sum = nums[0]
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum`
      },
      {
        id: 15,
        title: '3Sum',
        description: 'Find all unique triplets in the array which gives the sum of zero.',
        solution: `def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return res`
      },
      {
        id: 56,
        title: 'Merge Intervals',
        description: 'Merge overlapping intervals.',
        solution: `def merge(intervals):
    intervals.sort()
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged`
      },
      {
        id: 11,
        title: 'Container With Most Water',
        description: 'Find the maximum water that can be trapped between two lines.',
        solution: `def maxArea(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        max_water = max(max_water, min(height[left], height[right]) * (right - left))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water`
      },
      {
        id: 48,
        title: 'Rotate Image',
        description: 'Rotate a matrix 90 degrees clockwise.',
        solution: `def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()`
      }
    ],
    'Strings': [
      {
        id: 20,
        title: 'Valid Parentheses',
        description: 'Determine if the input string has valid parentheses.',
        solution: `def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in mapping:
            top = stack.pop() if stack else '#'
            if mapping[c] != top:
                return False
        else:
            stack.append(c)
    return not stack`
      },
      {
        id: 125,
        title: 'Valid Palindrome',
        description: 'Check if a string is a palindrome, ignoring non-alphanumeric characters and case.',
        solution: `def isPalindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True`
      },
      {
        id: 242,
        title: 'Valid Anagram',
        description: 'Determine if two strings are anagrams of each other.',
        solution: `def isAnagram(s, t):
    if len(s) != len(t):
        return False
    count = [0] * 26
    for c in s:
        count[ord(c) - ord('a')] += 1
    for c in t:
        count[ord(c) - ord('a')] -= 1
        if count[ord(c) - ord('a')] < 0:
            return False
    return True`
      },
      {
        id: 49,
        title: 'Group Anagrams',
        description: 'Group anagrams together from a list of strings.',
        solution: `from collections import defaultdict

def groupAnagrams(strs):
    anagram_map = defaultdict(list)
    for s in strs:
        sorted_str = ''.join(sorted(s))
        anagram_map[sorted_str].append(s)
    return list(anagram_map.values())`
      },
      {
        id: 5,
        title: 'Longest Palindromic Substring',
        description: 'Find the longest palindromic substring in a given string.',
        solution: `def longestPalindrome(s):
    n = len(s)
    if n == 0: return ""
    start = 0
    max_len = 1
    dp = [[False] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_len:
                        max_len = length
                        start = i
    return s[start:start + max_len]`
      },
      {
        id: 76,
        title: 'Minimum Window Substring',
        description: 'Find the minimum window in string s which contains all characters of string t.',
        solution: `from collections import Counter

def minWindow(s, t):
    if not s or not t: return ""
    freq = Counter(t)
    required = len(freq)
    left = 0
    min_len = float('inf')
    min_start = 0
    formed = 0
    window_counts = {}
    
    for right, char in enumerate(s):
        window_counts[char] = window_counts.get(char, 0) + 1
        if char in freq and window_counts[char] == freq[char]:
            formed += 1
        
        while left <= right and formed == required:
            char = s[left]
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            window_counts[char] -= 1
            if char in freq and window_counts[char] < freq[char]:
                formed -= 1
            left += 1
    
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]`
      },
      {
        id: 28,
        title: 'Find the Index of the First Occurrence',
        description: 'Implement strStr() - find substring in a string.',
        solution: `def strStr(haystack, needle):
    if not needle:
        return 0
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1`
      },
      {
        id: 443,
        title: 'String Compression',
        description: 'Compress a string by replacing repeated characters with character and count.',
        solution: `def compress(chars):
    index = 0
    i = 0
    n = len(chars)
    while i < n:
        char = chars[i]
        count = 0
        while i < n and chars[i] == char:
            i += 1
            count += 1
        chars[index] = char
        index += 1
        if count > 1:
            for digit in str(count):
                chars[index] = digit
                index += 1
    return index`
      },
      {
        id: 14,
        title: 'Longest Common Prefix',
        description: 'Find the longest common prefix string amongst an array of strings.',
        solution: `def longestCommonPrefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for i in range(1, len(strs)):
        while not strs[i].startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix`
      },
      {
        id: 459,
        title: 'Repeated Substring Pattern',
        description: 'Check if a string can be constructed by repeating a substring.',
        solution: `def repeatedSubstringPattern(s):
    doubled = s + s
    return s in doubled[1:-1]`
      }
    ],
    'Linked Lists': [
      {
        id: 206,
        title: 'Reverse Linked List',
        description: 'Reverse a singly linked list.',
        solution: `def reverseList(head):
    prev = None
    while head:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev`
      },
      {
        id: 21,
        title: 'Merge Two Sorted Lists',
        description: 'Merge two sorted linked lists into one sorted list.',
        solution: `def mergeTwoLists(list1, list2):
    if not list1:
        return list2
    if not list2:
        return list1
    if list1.val < list2.val:
        list1.next = mergeTwoLists(list1.next, list2)
        return list1
    else:
        list2.next = mergeTwoLists(list1, list2.next)
        return list2`
      },
      {
        id: 19,
        title: 'Remove Nth Node From End of List',
        description: 'Remove the nth node from the end of the list.',
        solution: `def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    slow = fast = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next`
      },
      {
        id: 141,
        title: 'Linked List Cycle',
        description: 'Determine if a linked list has a cycle.',
        solution: `def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`
      },
      {
        id: 2,
        title: 'Add Two Numbers',
        description: 'Add two numbers represented by linked lists.',
        solution: `def addTwoNumbers(l1, l2):
    dummy = ListNode()
    current = dummy
    carry = 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        total = val1 + val2 + carry
        carry = total // 10
        current.next = ListNode(total % 10)
        current = current.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    return dummy.next`
      },
      {
        id: 160,
        title: 'Intersection of Two Linked Lists',
        description: 'Find the intersection node of two linked lists.',
        solution: `def getIntersectionNode(headA, headB):
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a`
      },
      {
        id: 234,
        title: 'Palindrome Linked List',
        description: 'Check if a linked list is a palindrome.',
        solution: `def isPalindrome(head):
    slow = fast = head
    prev = None
    while fast and fast.next:
        fast = fast.next.next
        temp = slow.next
        slow.next = prev
        prev = slow
        slow = temp
    if fast:
        slow = slow.next
    while slow:
        if slow.val != prev.val:
            return False
        slow = slow.next
        prev = prev.next
    return True`
      },
      {
        id: 25,
        title: 'Reverse Nodes in k-Group',
        description: 'Reverse nodes in k-group in a linked list.',
        solution: `def reverseKGroup(head, k):
    def has_k_nodes(node, k):
        for _ in range(k):
            if not node:
                return False
            node = node.next
        return True

    if not has_k_nodes(head, k):
        return head

    prev = None
    current = head
    for _ in range(k):
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    head.next = reverseKGroup(current, k)
    return prev`
      }
    ],
    'Stacks and Queues': [
      {
        id: 232,
        title: 'Implement Queue using Stacks',
        description: 'Implement a queue using two stacks.',
        solution: `class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        self.s1.append(x)

    def pop(self):
        self.peek()
        return self.s2.pop()

    def peek(self):
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]

    def empty(self):
        return not self.s1 and not self.s2`
      },
      {
        id: 225,
        title: 'Implement Stack using Queues',
        description: 'Implement a stack using two queues.',
        solution: `from collections import deque

class MyStack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        return self.q1.popleft()

    def top(self):
        return self.q1[0]

    def empty(self):
        return len(self.q1) == 0`
      },
      {
        id: 155,
        title: 'Min Stack',
        description: 'Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.',
        solution: `class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]`
      },
      {
        id: 739,
        title: 'Daily Temperatures',
        description: 'Given a list of daily temperatures, return a list of days to wait for warmer temperature.',
        solution: `def dailyTemperatures(temperatures):
    stack = []
    res = [0] * len(temperatures)
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res`
      },
      {
        id: 150,
        title: 'Evaluate Reverse Polish Notation',
        description: 'Evaluate the value of an arithmetic expression in Reverse Polish Notation.',
        solution: `def evalRPN(tokens):
    stack = []
    for token in tokens:
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]`
      },
      {
        id: 496,
        title: 'Next Greater Element I',
        description: 'Find the next greater element for each element in nums1 from nums2.',
        solution: `def nextGreaterElement(nums1, nums2):
    next_greater = {}
    stack = []
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    return [next_greater.get(num, -1) for num in nums1]`
      },
      {
        id: 503,
        title: 'Next Greater Element II',
        description: 'Find the next greater element for each element in a circular array.',
        solution: `def nextGreaterElements(nums):
    n = len(nums)
    res = [-1] * n
    stack = []
    for i in range(2 * n):
        while stack and nums[stack[-1]] < nums[i % n]:
            res[stack.pop()] = nums[i % n]
        if i < n:
            stack.append(i)
    return res`
      },
      {
        id: 622,
        title: 'Circular Queue',
        description: 'Design a circular queue.',
        solution: `class MyCircularQueue:
    def __init__(self, k):
        self.queue = [0] * k
        self.head = self.tail = -1
        self.size = k

    def enQueue(self, value):
        if self.isFull():
            return False
        if self.isEmpty():
            self.head = 0
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = value
        return True

    def deQueue(self):
        if self.isEmpty():
            return False
        if self.head == self.tail:
            self.head = self.tail = -1
        else:
            self.head = (self.head + 1) % self.size
        return True

    def Front(self):
        return -1 if self.isEmpty() else self.queue[self.head]

    def Rear(self):
        return -1 if self.isEmpty() else self.queue[self.tail]

    def isEmpty(self):
        return self.head == -1

    def isFull(self):
        return (self.tail + 1) % self.size == self.head`
      }
    ],
    'Binary Search': [
      {
        id: 704,
        title: 'Binary Search',
        description: 'Implement binary search algorithm.',
        solution: `def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1`
      },
      {
        id: 34,
        title: 'Find First and Last Position',
        description: 'Find the starting and ending position of a target value in a sorted array.',
        solution: `def searchRange(nums, target):
    def findLeft():
        left, right = 0, len(nums) - 1
        index = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                index = mid
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return index

    def findRight():
        left, right = 0, len(nums) - 1
        index = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                index = mid
                left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return index

    return [findLeft(), findRight()]`
      },
      {
        id: 74,
        title: 'Search a 2D Matrix',
        description: 'Search for a target value in a sorted 2D matrix.',
        solution: `def searchMatrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    while left <= right:
        mid = (left + right) // 2
        mid_val = matrix[mid // cols][mid % cols]
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    return False`
      },
      {
        id: 33,
        title: 'Search in Rotated Sorted Array',
        description: 'Search for a target value in a rotated sorted array.',
        solution: `def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1`
      },
      {
        id: 81,
        title: 'Search in Rotated Sorted Array II',
        description: 'Search for a target value in a rotated sorted array with duplicates.',
        solution: `def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return False`
      },
      {
        id: 162,
        title: 'Find Peak Element',
        description: 'Find a peak element in an array.',
        solution: `def findPeakElement(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left`
      }
    ],
    'Trees': [
      {
        id: 104,
        title: 'Maximum Depth of Binary Tree',
        description: 'Find the maximum depth of a binary tree.',
        solution: `def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))`
      },
      {
        id: 100,
        title: 'Same Tree',
        description: 'Check if two binary trees are the same.',
        solution: `def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)`
      },
      {
        id: 101,
        title: 'Symmetric Tree',
        description: 'Check if a binary tree is symmetric.',
        solution: `def isSymmetric(root):
    def isMirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2 or t1.val != t2.val:
            return False
        return isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)
    return isMirror(root, root)`
      },
      {
        id: 144,
        title: 'Binary Tree Preorder Traversal',
        description: 'Return the preorder traversal of a binary tree.',
        solution: `def preorderTraversal(root):
    if not root:
        return []
    stack, result = [root], []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result`
      },
      {
        id: 94,
        title: 'Binary Tree Inorder Traversal',
        description: 'Return the inorder traversal of a binary tree.',
        solution: `def inorderTraversal(root):
    result = []
    stack = []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result`
      },
      {
        id: 145,
        title: 'Binary Tree Postorder Traversal',
        description: 'Return the postorder traversal of a binary tree.',
        solution: `def postorderTraversal(root):
    if not root:
        return []
    stack, result = [], []
    last_visited = None
    curr = root
    while curr or stack:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            peek = stack[-1]
            if peek.right and last_visited != peek.right:
                curr = peek.right
            else:
                result.append(peek.val)
                last_visited = stack.pop()
    return result`
      },
      {
        id: 102,
        title: 'Binary Tree Level Order Traversal',
        description: 'Return the level order traversal of a binary tree.',
        solution: `from collections import deque

def levelOrder(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result`
      },
      {
        id: 110,
        title: 'Balanced Binary Tree',
        description: 'Check if a binary tree is height-balanced.',
        solution: `def isBalanced(root):
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)
    return height(root) != -1`
      }
    ],
    'Recursion and Backtracking': [
      {
        id: 39,
        title: 'Combination Sum',
        description: 'Find all combinations that sum up to target.',
        solution: `def combinationSum(candidates, target):
    def backtrack(start, current, result):
        if target == sum(current):
            result.append(current[:])
            return
        for i in range(start, len(candidates)):
            if sum(current) + candidates[i] > target:
                continue
            current.append(candidates[i])
            backtrack(i, current, result)
            current.pop()
    result = []
    backtrack(0, [], result)
    return result`
      },
      {
        id: 46,
        title: 'Permutations',
        description: 'Generate all permutations of a list of numbers.',
        solution: `def permute(nums):
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    result = []
    backtrack(0)
    return result`
      },
      {
        id: 78,
        title: 'Subsets',
        description: 'Generate all possible subsets of a set.',
        solution: `def subsets(nums):
    def backtrack(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    result = []
    backtrack(0, [])
    return result`
      },
      {
        id: 51,
        title: 'N-Queens',
        description: 'Solve the N-Queens problem.',
        solution: `def solveNQueens(n):
    def isSafe(row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        return True

    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if isSafe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    result = []
    board = [['.'] * n for _ in range(n)]
    backtrack(0)
    return result`
      },
      {
        id: 17,
        title: 'Letter Combinations of a Phone Number',
        description: 'Generate all possible letter combinations from a phone number.',
        solution: `def letterCombinations(digits):
    if not digits:
        return []
    mapping = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
    result = []
    def backtrack(index, current):
        if index == len(digits):
            result.append(current)
            return
        for ch in mapping[int(digits[index])]:
            backtrack(index + 1, current + ch)
    backtrack(0, "")
    return result`
      },
      {
        id: 90,
        title: 'Subsets II',
        description: 'Generate all possible subsets with duplicates.',
        solution: `def subsetsWithDup(nums):
    def backtrack(start, current):
        result.append(current[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    nums.sort()
    result = []
    backtrack(0, [])
    return result`
      },
      {
        id: 37,
        title: 'Sudoku Solver',
        description: 'Solve a Sudoku puzzle.',
        solution: `def solveSudoku(board):
    def isValid(row, col, c):
        for i in range(9):
            if board[i][col] == c or board[row][i] == c or \\
               board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == c:
                return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for c in '123456789':
                        if isValid(i, j, c):
                            board[i][j] = c
                            if solve():
                                return True
                            board[i][j] = '.'
                    return False
        return True

    solve()`
      }
    ],
    'Dynamic Programming': [
      {
        id: 70,
        title: 'Climbing Stairs',
        description: 'Count ways to climb stairs taking 1 or 2 steps at a time.',
        solution: `def climbStairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b`
      },
      {
        id: 198,
        title: 'House Robber',
        description: 'Maximize amount robbed without robbing adjacent houses.',
        solution: `def rob(nums):
    prev1 = prev2 = 0
    for num in nums:
        prev1, prev2 = max(prev1, prev2 + num), prev1
    return prev1`
      },
      {
        id: 322,
        title: 'Coin Change',
        description: 'Find minimum number of coins to make up an amount.',
        solution: `def coinChange(coins, amount):
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)
    return dp[amount] if dp[amount] <= amount else -1`
      },
      {
        id: 300,
        title: 'Longest Increasing Subsequence',
        description: 'Find the length of the longest increasing subsequence.',
        solution: `def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0`
      },
      {
        id: 1143,
        title: 'Longest Common Subsequence',
        description: 'Find the length of the longest common subsequence.',
        solution: `def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]`
      },
      {
        id: 62,
        title: 'Unique Paths',
        description: 'Count unique paths from top-left to bottom-right in a grid.',
        solution: `def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]`
      },
      {
        id: 5,
        title: 'Longest Palindromic Substring',
        description: 'Find the longest palindromic substring (DP version).',
        solution: `def longestPalindrome(s):
    n = len(s)
    if n == 0: return ""
    start = 0
    max_len = 1
    dp = [[False] * n for _ in range(n)]
    
    for i in range(n):
        dp[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or dp[i + 1][j - 1]:
                    dp[i][j] = True
                    if length > max_len:
                        max_len = length
                        start = i
    return s[start:start + max_len]`
      },
      {
        id: 718,
        title: 'Maximum Length of Repeated Subarray',
        description: 'Find the maximum length of a subarray that appears in both arrays.',
        solution: `def findLength(nums1, nums2):
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
    return max_len`
      },
      {
        id: 416,
        title: 'Partition Equal Subset Sum',
        description: 'Determine if an array can be partitioned into two subsets with equal sum.',
        solution: `def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]`
      },
      {
        id: 53,
        title: 'Maximum Subarray',
        description: 'Find the contiguous subarray with the largest sum (Kadane\'s algorithm).',
        solution: `def maxSubArray(nums):
    max_sum = current_sum = nums[0]
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    return max_sum`
      }
    ],
    'Graphs': [
      {
        id: 133,
        title: 'Clone Graph',
        description: 'Clone an undirected graph.',
        solution: `class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    if not node:
        return None
    visited = {}
    from collections import deque
    q = deque([node])
    visited[node] = Node(node.val)
    while q:
        curr = q.popleft()
        for neighbor in curr.neighbors:
            if neighbor not in visited:
                visited[neighbor] = Node(neighbor.val)
                q.append(neighbor)
            visited[curr].neighbors.append(visited[neighbor])
    return visited[node]`
      },
      {
        id: 200,
        title: 'Number of Islands',
        description: 'Count the number of islands in a 2D grid.',
        solution: `def numIslands(grid):
    if not grid:
        return 0
    def dfs(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count`
      },
      {
        id: 207,
        title: 'Course Schedule',
        description: 'Determine if you can finish all courses given prerequisites.',
        solution: `from collections import deque

def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses
    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1
    q = deque([i for i in range(numCourses) if indegree[i] == 0])
    count = 0
    while q:
        course = q.popleft()
        count += 1
        for neighbor in graph[course]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    return count == numCourses`
      },
      {
        id: 785,
        title: 'Is Graph Bipartite?',
        description: 'Check if a graph is bipartite.',
        solution: `def isBipartite(graph):
    n = len(graph)
    color = [-1] * n
    for i in range(n):
        if color[i] != -1:
            continue
        q = deque([i])
        color[i] = 0
        while q:
            node = q.popleft()
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    color[neighbor] = 1 - color[node]
                    q.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True`
      },
      {
        id: 994,
        title: 'Rotting Oranges',
        description: 'Find minimum time to rot all oranges.',
        solution: `from collections import deque

def orangesRotting(grid):
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    if fresh == 0:
        return 0
    minutes = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while q and fresh > 0:
        for _ in range(len(q)):
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    grid[nx][ny] = 2
                    fresh -= 1
                    q.append((nx, ny))
        minutes += 1
    return minutes if fresh == 0 else -1`
      },
      {
        id: 323,
        title: 'Number of Connected Components',
        description: 'Count the number of connected components in an undirected graph.',
        solution: `def countComponents(n, edges):
    graph = [[] for _ in range(n)]
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    visited = [False] * n
    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    count = 0
    for i in range(n):
        if not visited[i]:
            count += 1
            dfs(i)
    return count`
      }
    ],
    'Bit Manipulation': [
      {
        id: 136,
        title: 'Single Number',
        description: 'Find the single number that appears only once in an array.',
        solution: `def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result`
      },
      {
        id: 190,
        title: 'Reverse Bits',
        description: 'Reverse the bits of a 32-bit unsigned integer.',
        solution: `def reverseBits(n):
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result`
      },
      {
        id: 191,
        title: 'Number of 1 Bits',
        description: 'Count the number of 1 bits in an unsigned integer.',
        solution: `def hammingWeight(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count`
      },
      {
        id: 268,
        title: 'Missing Number',
        description: 'Find the missing number in an array containing n distinct numbers from 0 to n.',
        solution: `def missingNumber(nums):
    n = len(nums)
    total_xor = 0
    for i in range(n + 1):
        total_xor ^= i
    array_xor = 0
    for num in nums:
        array_xor ^= num
    return total_xor ^ array_xor`
      }
    ]
  };

  const categories = Object.keys(dsaProblems);
  
  // Filter problems based on active category and search term
  const filteredProblems = dsaProblems[activeCategory]?.filter(problem => 
    problem.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    problem.description.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  useEffect(() => {
    // Set dark mode based on user preference
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDark);
  }, []);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('Code copied to clipboard!');
  };

  return (
    <div className={`min-h-screen transition-colors duration-300 ${isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      {/* Header */}
      <header className={`py-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`}>
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div className="mb-4 md:mb-0">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                75 DSA Questions from LeetCode
              </h1>
              <p className={`mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                BY: Narayanam Mahanandhan | Python Solutions
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search problems..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={`pl-10 pr-4 py-2 rounded-lg w-64 focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                    isDarkMode ? 'bg-gray-700 text-white border-gray-600' : 'bg-white text-gray-900 border-gray-300'
                  } border`}
                />
                <svg className="absolute left-3 top-2.5 h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <button
                onClick={toggleDarkMode}
                className={`p-2 rounded-full ${isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors`}
              >
                {isDarkMode ? (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar */}
          <div className={`w-full lg:w-64 flex-shrink-0 ${isDarkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-4`}>
            <h2 className="text-xl font-semibold mb-4">Categories</h2>
            <nav className="space-y-2">
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setActiveCategory(category)}
                  className={`w-full text-left px-4 py-2 rounded-md transition-colors ${
                    activeCategory === category
                      ? 'bg-blue-500 text-white'
                      : isDarkMode
                      ? 'hover:bg-gray-700 text-gray-200'
                      : 'hover:bg-gray-100 text-gray-700'
                  }`}
                >
                  {category}
                </button>
              ))}
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1">
            {selectedProblem ? (
              <div className={`rounded-lg shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h2 className="text-2xl font-bold mb-2">{selectedProblem.title}</h2>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      LeetCode #{selectedProblem.id}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedProblem(null)}
                    className={`p-2 rounded-full ${isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-2">Problem Description</h3>
                  <p className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>
                    {selectedProblem.description}
                  </p>
                </div>
                
                <div className="mb-6">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-lg font-semibold">Python Solution</h3>
                    <button
                      onClick={() => copyToClipboard(selectedProblem.solution)}
                      className={`px-4 py-2 rounded-md text-sm font-medium ${
                        isDarkMode 
                          ? 'bg-blue-600 hover:bg-blue-700 text-white' 
                          : 'bg-blue-500 hover:bg-blue-600 text-white'
                      } transition-colors`}
                    >
                      Copy Code
                    </button>
                  </div>
                  <div className={`p-4 rounded-md overflow-x-auto ${
                    isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-gray-100 text-gray-900'
                  }`}>
                    <pre className="text-sm font-mono whitespace-pre-wrap">
                      {selectedProblem.solution}
                    </pre>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold">{activeCategory}</h2>
                  <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                    {filteredProblems.length} problems found
                  </p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {filteredProblems.map(problem => (
                    <div
                      key={problem.id}
                      onClick={() => setSelectedProblem(problem)}
                      className={`rounded-lg p-5 cursor-pointer transition-all transform hover:scale-105 ${
                        isDarkMode 
                          ? 'bg-gray-800 hover:bg-gray-700 border border-gray-700' 
                          : 'bg-white hover:bg-gray-50 shadow-md hover:shadow-lg'
                      }`}
                    >
                      <div className="flex justify-between items-start">
                        <h3 className="text-lg font-semibold">{problem.title}</h3>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          isDarkMode 
                            ? 'bg-blue-900 text-blue-200' 
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          #{problem.id}
                        </span>
                      </div>
                      <p className={`mt-3 text-sm line-clamp-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                        {problem.description}
                      </p>
                      <div className="mt-4 flex items-center text-sm">
                        <span className={isDarkMode ? 'text-blue-400' : 'text-blue-600'}>
                          Click to view solution
                        </span>
                        <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </div>
                  ))}
                </div>
                
                {filteredProblems.length === 0 && (
                  <div className={`text-center py-12 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`}>
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.291-.947-5.824-2.379a7.962 7.962 0 010-5.242 7.962 7.962 0 015.824-2.379 7.962 7.962 0 015.824 2.379 7.962 7.962 0 010 5.242A7.962 7.962 0 0112 15z" />
                    </svg>
                    <h3 className="mt-4 text-lg font-medium">No problems found</h3>
                    <p className={`mt-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                      Try adjusting your search term or select a different category.
                    </p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className={`mt-12 py-6 ${isDarkMode ? 'bg-gray-800 text-gray-300' : 'bg-gray-100 text-gray-600'}`}>
        <div className="container mx-auto px-4 text-center">
          <p>© 2024 DSA Solutions Collection. All Python solutions converted from C++.</p>
          <p className="mt-2 text-sm">Perfect for interview preparation and coding practice.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
