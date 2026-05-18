---
title: 'An Interesting Encoding Problem'
description: 'A coin-guessing game where winning rounds and transmitting information fight each other.'
pubDate: 'May 20 2025'
heroImage: '../../assets/blog-placeholder-3.jpg'
---

One of my tutoring students once asked me a game-theoretic math problem about information encoding. I managed to solve only the first two parts after a long struggle. Later, after looking through several sources, I finally understood the rest parts. So, let's dive in.

## The Problem

A teacher plays a coin-guessing game with two students, Alice and Bob. The teacher is one team; Alice and Bob are the other.

The game consists of several rounds. In each round, the teacher, Alice, and Bob each secretly choose one side of a coin, then reveal their choices simultaneously. If all three coins show heads, or all three show tails, then the students win that round.

Once the game starts, Alice and Bob may no longer communicate in any form. The only information available to them is the complete revealed result of each previous round.

This rule is obviously unfavorable to the students, and by itself it is mostly luck. So we add one extra rule: after the game starts but before the first round, the teacher secretly tells Alice the teacher's entire future sequence of coin choices. The teacher must then follow that sequence honestly. In other words, Alice knows every future move of the teacher from the beginning, but the game has already started, so she cannot directly tell Bob.

Before the game begins, Alice and Bob have as much time as they want to agree on a strategy. Design strategies that guarantee the students can:

1. Win 1 round out of 2.
2. Win 3 rounds out of 5.
3. Win 6 rounds out of 9.
4. Win 9 rounds out of 13.

## Reduction

The core tension is that Alice must balance two goals: scoring points now, and using some rounds to hint at how Bob should score points later.

Let $A_i, B_i, C_i$ denote the choices of Alice, Bob, and the teacher in round $i$, where $1 \leq i \leq N$.

Alice's behavior may depend on the teacher's entire sequence:

$$
\vec{A}=F(\vec{C})
$$

Bob's behavior in round $i$ may depend only on the revealed history and the strategy agreed in advance:

$$
B_i=G(A_1,\dots,A_{i-1},C_1,\dots,C_{i-1})
$$

The goal is to maximize

$$
\min_{\vec{C}\in\{0,1\}^N}{\sum_{i=1}^N [A_i=B_i=C_i]}.
$$

The functions $F$ and $G$ are not required to be linear. Even for $F$, if it were linear, then

$$
\vec{A}_N=F_{N\times N} \vec{C}_N,
$$

there would be $2^{N^2}$ possible matrices. In the general case, $F$ is a map from $2^N$ possible inputs to $2^N$ possible outputs, so the number of possibilities is

$$
(2^N)^{2^N}=2^{N\cdot 2^N}.
$$

That is far beyond a pleasant brute-force search.

## One Constructive Solution

The following strategy is adapted from [蓬莱の魚の形](https://space.bilibili.com/208604). The original source is here: https://www.bilibili.com/read/readlist/rl305703

The two central demands, "win this round" and "transmit information", naturally compete with each other.

### Winning 3 Out of 5

Split the 5 rounds into groups of

$$
1+3+1.
$$

In the first group, Alice tells Bob which side appears more often in the middle group of 3 rounds. Bob then plays only that side throughout the middle group.

If all three teacher choices in the middle group are the same, Bob wins all 3 of them. The last round no longer matters, and the score is

$$
0+3+0=3.
$$

If the middle group is not uniform, Bob wins 2 and loses 1. The necessarily lost round can be used by Alice to tell Bob the teacher's choice in the final round, so the score is

$$
0+2+1=3.
$$

Why can this guarantee more than half the rounds? The key is that the strategy does **not** preassign fixed rounds for communication and fixed rounds for winning.

To win 2 rounds in the middle group of 3, Alice does not need to send 2 bits describing the exact pattern. She only sends 1 bit: the majority side. This is the first important trick:

**Transmit a coarse description of a range, rather than an exact description of a specific future state.**

This idea already generalizes easily: in $3n+2$ rounds, the students can guarantee $2n+1$ wins.

### Extracting Extra Information

There is more information hidden in the pattern of deliberate losses.

1. Alice may deliberately send the minority side of the middle group. Then Bob wins 1 and loses 2, and those two lost rounds can transmit 2 bits.
2. Alice may send the majority side, but deliberately make a mistake in a round that should have been won. Since there are two would-be winning rounds, the earlier or later deliberate mistake transmits 1 bit. Together with the naturally lost round, this also gives 2 bits.

So we have two useful principles:

1. Create more necessarily lost rounds; each such round can transmit 1 bit.
2. Deliberately give up a round that would otherwise be won; the position of the deliberate loss carries information.

Even the choice of which strategy is being used can itself carry information.

For a $1+3$ block, Alice can arrange the following outputs:

1. Tell Bob that the future is branch 1, win 1 round, and transmit 2 bits.
2. Tell Bob that the future is branch 2, win 1 round, and transmit 2 bits.
3. Tell Bob that the future is branch 3, win 2 rounds, and transmit 1 bit.

### Winning 6 Out of 9

Split the 9 rounds as

$$
1+3+4+1.
$$

First inspect the shape of the 3-round group.

If those 3 rounds are all the same side, Alice uses the first round to indicate that side, and Bob wins all 3. The remaining 5 rounds are exactly the previous "win 3 out of 5" subproblem. Since Bob sees that all 3 middle rounds were won, he also knows immediately which strategy to switch to.

If the 3-round group is not uniform, Alice next inspects the shape of the 4-round group. Write the two coin sides as $a$ and $b$.

**Branch 1: the 4-round group is `abba` or `baab`.**

Alice uses the first group to tell Bob the minority side in the 3-round group. Bob plays only that side, so the 3-round group gives 1 win and 2 losses. Seeing that Alice deliberately sent the minority side, Bob recognizes branch 1 and knows that the 4-round group is either `abba` or `baab`.

The two necessarily lost rounds transmit the remaining details. In the first lost round, Alice uses $a$ to mean `abba` and $b$ to mean `baab`. In the second lost round, Alice uses $a$ or $b$ to indicate the final 1-round group. Then Bob wins the last 5 rounds, for a total of

$$
0+1+4+1=6.
$$

**Branch 2: the 4-round group is `abab` or `baba`.**

Alice uses the first group to tell Bob the majority side in the 3-round group. Bob would normally win 2 and lose 1, but Alice deliberately makes one of the winning rounds fail. Bob sees the deliberate mistake and recognizes branch 2.

If the earlier possible winning round is deliberately lost, it means `abab`; if the later one is lost, it means `baba`. The naturally lost round then tells Bob the final 1-round choice. Again the last 5 rounds are won, so the total is

$$
0+1+4+1=6.
$$

**Branch 3: the 4-round group is unbalanced, `aabb`, or `bbaa`.**

Alice tells Bob the majority side in the 3-round group. Bob plays that side and wins 2, loses 1. Seeing the normal 2-win, 1-loss pattern, Bob recognizes branch 3.

The naturally lost round now depends on whether the 4-round group is balanced.

If the 4-round group is unbalanced, Alice uses the lost round to tell Bob the majority side of that group. Bob plays that side throughout the 4-round group. If the pattern is $4:0$, Bob wins all 4 and the last round no longer matters:

$$
0+2+4=6.
$$

If the pattern is $3:1$, Bob wins 3, and the one necessarily lost round in this group tells him the final 1-round choice:

$$
0+2+3+1=6.
$$

If the 4-round group is balanced, it is either `aabb` or `bbaa`. Alice uses the lost round in the 3-round group to tell Bob the first side of the 4-round group: $a$ for `aabb`, $b$ for `bbaa`. Bob does not know yet that the group is balanced, so he first behaves as if it were unbalanced. This makes the first two rounds correct.

Alice then deliberately loses one of those first two rounds to signal that the group is actually balanced. Once Bob sees that deliberate mistake within branch 3, he knows the last two rounds should switch side, and he wins them. The position of the deliberate mistake also indicates the final 1-round choice. This gives

$$
0+2+3+1=6.
$$

### Modularizing the Strategy

So far we have two observations:

1. Every necessarily lost round can transmit 1 bit.
2. If the number of wins required is fixed, then the position of deliberate losses among otherwise winnable rounds can transmit $\binom{n}{r}$ possibilities, or $\log \binom{n}{r}$ bits.

Treat the initial $1+3$ part as a `begin()` module. Its possible outputs are:

1. Branch 1: win 1 round and obtain 2 bits of information.
2. Branch 2: win 1 round and obtain 2 bits of information.
3. Branch 3: win 2 rounds and obtain 1 bit of information.

Treat the final $4+1$ part as an `end()` module. Its possible inputs are:

1. Bob must know it is branch 1; use 2 bits and win 5 rounds.
2. Bob must know it is branch 2; use 2 bits and win 5 rounds.
3. Bob must know it is branch 3; use 1 bit and win 4 rounds.

The remaining task is to construct a `mid()` module. For each input branch, its three possible output branches must be distinguishable to Bob. Alice must provide enough information for Bob to know exactly which output branch he is in.

### Winning 9 Out of 13, and the $4n+1$ Pattern

The intended modular shape is

$$
1 \to 3 \to 4 \to 4 \to \cdots \to 4 \to 1.
$$

Below is the case analysis for one 4-round `mid()` block.

**Input branch 1: the `mid()` shape is `abba` or `baab`.**

The input provides 2 bits, but only 1 bit is needed to distinguish `abba` from `baab`.

For output branch 3, use 1 bit, win all 4 rounds, and pass the unused 1 bit onward as the output information. The all-win pattern tells Bob that the output is branch 3.

For output branch 1 or 2, first use 1 bit to distinguish `abba` from `baab`, then deliberately lose one of the 4 positions. This wins 3 rounds. The chosen lost position has 4 possibilities; combined with the unused input bit, this gives 3 bits. Use 1 bit to tell Bob whether the output is branch 1 or branch 2, and leave the other 2 bits as the output information. The 3-win, 1-loss pattern tells Bob that the output is branch 1 or 2.

**Input branch 2: the `mid()` shape is `abab` or `baba`.**

This is essentially the same as input branch 1. The input provides 2 bits; use 1 bit to distinguish `abab` from `baba`.

For output branch 3, win all 4 rounds and pass the unused 1 bit onward. The all-win pattern tells Bob that the output is branch 3.

For output branch 1 or 2, deliberately lose one of the 4 positions. This wins 3 rounds and produces 2 bits from the position, plus 1 unused input bit, so again there are 3 bits available for the output.

**Input branch 3: the `mid()` shape is `aabb`, `bbaa`, or unbalanced.**

The input provides 1 bit.

If all 4 rounds have the same side, then the 1 input bit can make Bob win all 4 rounds directly. Together with the 2 wins from `begin(out3)`, we already have 6 wins and can reduce the rest to the "win 3 out of 5" subproblem. This special case is useful, but it is awkward for the general $4n+1$ construction, so I set it aside here and focus on the remaining cases.

There are two main cases.

**Case 1: the shape is `aabb` or `bbaa`.**

For output branch 3, use the input bit to distinguish `aabb` from `bbaa`. Bob initially assumes the group is unbalanced and plays only $a$ or only $b$. Alice deliberately loses one of positions 1 or 2, telling Bob that the group is actually balanced. Bob then switches for positions 3 and 4. This wins 3 rounds. The deliberately lost position, either 1 or 2, transmits 1 bit as the output information. The 3-win, 1-loss pattern tells Bob that the output is branch 3.

For output branch 1, again use the input bit to distinguish `aabb` from `bbaa`. Alice deliberately loses one of positions 1 or 2 to reveal the balanced shape, and also deliberately loses one of positions 3 or 4. This wins 2 rounds. The two lost positions can be `13`, `14`, `23`, or `24`, giving 2 output bits. This pattern tells Bob that the output is branch 1.

For output branch 2, Alice makes positions 1 and 2 correct. Bob therefore does not realize that he should switch, so positions 3 and 4 are lost. The total is 2 wins. Although Bob has been assuming the unbalanced case, after seeing the teacher's complete pattern `aabb` or `bbaa`, he can still infer that this was the balanced case. The two necessarily lost rounds can transmit 2 output bits, and the pattern tells Bob that the output is branch 2.

**Case 2: the shape is a $3:1$ unbalanced pattern.**

For output branch 3, use the input bit to tell Bob which side is the majority. Bob plays only that side, wins 3 rounds, and loses 1. The necessarily lost round transmits 1 output bit. The 3-win, 1-loss pattern tells Bob that the output is branch 3.

For output branch 1 or 2, we need 2 wins, 2 losses, and 3 output bits.

First, Alice may use the input bit to indicate the majority side. Bob plays only that side. The naturally lost round gives Alice two choices, $a$ or $b$, and Alice also deliberately loses one of the remaining 3 rounds. This gives $2\cdot 3=6$ distinguishable possibilities. Depending on the exact pattern, Bob may temporarily mistake the group for a balanced one and switch in the last two positions, but Alice knows this and the strategy can account for it.

Second, Alice may instead use the input bit to indicate the minority side. This is the abnormal mode. Bob initially plays only that side.

If the minority side appears in the last two positions, then after losing the first two rounds Bob realizes this is the abnormal mode. The two necessarily lost rounds transmit 2 bits. One bit tells him whether the last two positions are `ab` or `ba`, making both later rounds winnable; the other bit remains as free output information.

If the minority side appears in the first two positions, then after three rounds Bob sees one win and two losses, and recognizes the abnormal mode. The final round can then be won. The two necessarily lost rounds could transmit 2 bits, giving 4 possibilities, but we only need 2 of them.

Combining the 6 possibilities from the majority-side mode with 2 possibilities from the abnormal mode gives 8 cases, hence 3 output bits for branch 1 or 2. The 2-win, 2-loss pattern tells Bob that the output is branch 1 or 2.

## Simulation Notes, Half Done

I also started writing a simulator for the strategy. This is still incomplete, but I keep it here because it records the intended encoding logic.

```python
C = list(map(int, input().split()))
n = len(C)
assert n % 4 == 1 and n >= 9
                
def C_to_A_begin(C, t, info) :
    assert len(C) == 4
    a = sum(C[1:4])
    A = [0] * 4
    if a == 0 : return 0, [0] * 4
    if a == 3 : return 0, [1] * 4
    if a == 1 or a == 2:
        b = 2 - a
        id = C.index(b)
        if t == 1:
            A[0] = A[id] = b
            A[2 - (id >= 2)], A[3 - (id == 3)] = (info >> 1), (info & 1)
        if t == 2:
            A[0], A[id] = (b ^ 1), (info >> 1)
            A[2 - (id >= 2)], A[3 - (id == 3)] = b ^ (info & 1), b ^ (info & 1) ^ 1
        if t == 3:
            A = [b ^ 1] * 4
            A[id] = info
        return 1, A
    
def C_to_A_mid(M, t, info, tag):
    assert len(M) == 4
    m = M[0] * 8 + M[1] * 4 + M[2] * 2 + M[3]
    R = M.copy()
    T, INFO = 0, 0
    if tag : 
        t, A = C_to_A_begin(R, t, info)
    if m == 0b0110 or m == 0b1001 or m == 0b0101 or m == 0b1010:
        T = 2 - (M[0] ^ M[2])
        if t == 1 or t == 2:
            INFO = M[0] * 2 + t == 1
            R[info] ^= 1
        else :  INFO = M[0] * 2 + info
    else :
        T = 3
        s = sum(M)
        if m == 0b0011 or m == 0b1100:
            INFO = M[0]
            if t == 1 :
                R[info >> 1] ^= 1
                R[(info & 1) + 2] ^= 1
            if t == 2 : R[2], R[3] = (info >> 1), (info & 1)
            if t == 3 : R[info] ^= 1
        else :
            INFO = M[0]
            if t == 3 :

def C_to_A(C):
    b = C[-5] * 8 + C[-4] * 4 + C[-3] * 2 + C[-2]
    n = len(C)
    A = [0] * n
    A[-5:] = C[-5:]
    t, info = 0, 0
    if b == 0b0110 or b == 0b1001: t, info = 1, C[-5] * 2 + C[-1]
    elif b == 0b0101 or b == 0b1010: t, info = 2, C[-5] * 2 + C[-1]
    else :
        c = sum(C[-5:-1])
        t, info = 3, C[-5]
        if b == 0b0011 or b == 0b1100:
            if C[-1] == 0: A[-5] ^= 1
            else : A[-4] ^= 1
        elif c == 1 or c == 3: A[C[-5:-1].index(c == 1) - 5] = C[-1]
    for i in range((n - 9) // 4, 0, -1):
        A[i * 4:i * 4 + 4], t, info = C_to_A_mid(C[i * 4:i * 4 + 4], t, info, i > 1 and sum(C[i * 4 - 4:i * 4]) == 4)
```

## Related Literature, Half Done

[Oliver Gossner](https://scholar.google.fr/citations?user=WQs3TLcAAAAJ&hl=fr)

APA: Gossner, O., Hernández, P., & Neyman, A. (2006). Optimal use of communication resources. *Econometrica*, *74*(6), 1603-1636. https://doi.org/10.1111/j.1468-0262.2006.00720.x

Article page: https://cris.huji.ac.il/en/publications/optimal-use-of-communication-resources

Local archive: [Econometrica - 2006 - Gossner - Optimal Use of Communication Resources.pdf](/assets/papers/Econometrica - 2006 - Gossner - Optimal Use of Communication Resources.pdf)

Section 2.1, "Example: Coordination with Nature", presents this problem.

### Block Size 3

The target table is

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{17}{48} & \frac{1}{16} \\
1 & \frac{1}{48} & \frac{1}{16} \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{1}{16} & \frac{1}{48} \\
1 & \frac{1}{16} & \frac{17}{48} \\
\end{array}
$$

For example, in the state $(0,0,0)$:

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{5}{6} & 0 \\
1 & \frac{1}{6} & 0 \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & 0 & 0 \\
1 & 0 & 0 \\
\end{array}
$$

Averaging over $(0,0,0)$ and $(1,1,1)$ gives

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{5}{12} & 0 \\
1 & \frac{1}{12} & 0 \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & 0 & \frac{1}{12} \\
1 & 0 & \frac{5}{12} \\
\end{array}
$$

For $(0,0,1)$, $(0,1,0)$, or $(1,0,0)$:

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{2}{3} & 0 \\
1 & 0 & 0 \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{1}{6} & 0 \\
1 & \frac{1}{6} & 0 \\
\end{array}
$$

Averaging over

$$
(0,0,1),(0,1,0),(1,0,0),(1,1,0),(1,0,1),(0,1,1)
$$

gives

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{1}{3} & \frac{1}{12} \\
1 & 0 & \frac{1}{12} \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{1}{12} & 0 \\
1 & \frac{1}{12} & \frac{1}{3} \\
\end{array}
$$

The weighted average is

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{17}{48} & \frac{1}{16} \\
1 & \frac{1}{48} & \frac{1}{16} \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{1}{16} & \frac{1}{48} \\
1 & \frac{1}{16} & \frac{17}{48} \\
\end{array}
$$

### Information Constraint

$$
i=0: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{x}{2} & \frac{1-x}{6} \\
1 & \frac{1-x}{6} & \frac{1-x}{6} \\
\end{array}
\qquad
i=1: \quad
\begin{array}{c|cc}
 & 0 & 1 \\
\hline
0 & \frac{1-x}{6} & \frac{1-x}{6} \\
1 & \frac{1-x}{6} & \frac{x}{2} \\
\end{array}
$$

$$
H(P) = -\sum_{i,j} P(i,j) \log P(i,j)
$$

$$
H(i,j \mid k) = - \left[ y \log y + 3 \cdot \frac{1 - y}{3} \log \left( \frac{1 - y}{3} \right) \right] =H(y) + (1 - y)\log 3
$$
