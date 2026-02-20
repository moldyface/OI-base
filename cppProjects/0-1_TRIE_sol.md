# 0 - 1 TRIE

Prerequisites: TRIE data structure

# Problem
Given $n$ numbers, maximize $a$ xor $b$, where $a,b$ are elements in the list.

Time complexity: $O(n\log(n))$

Sol:

Use a 0 - 1 Trie to store each number.

For each number, extract its binary digits. From MSD to LSD choose greedily such that each digit is 1. 

E.g

If the number is $13 = 1101_2$, we should aim for a number like $1110010_2$.

So at every moment, if we can choose a number such that the current digit xor's to become $1$, then we should choose it

Why? because this digit has more influence than all digits below it. if this is equal to $0$, it doesnt matter if everything else is 1. it is still less than if it is $1$.