# A short proof that the sum of the first n odd numbers is a perfect square

## Abstract
We give a short induction proof that the sum of the first n odd positive integers equals n squared, and check it against direct computation for small n.

## Introduction
The identity 1 + 3 + 5 + ... + (2n - 1) = n^2 is classical. We record a self-contained proof for teaching use. No new result is claimed.

## Main result
Theorem 1. For every positive integer n, the sum S(n) = 1 + 3 + ... + (2n - 1) equals n^2.

Proof. For n = 1, S(1) = 1 = 1^2, so the claim holds. Suppose S(k) = k^2 for some positive integer k. Then S(k + 1) = S(k) + (2(k + 1) - 1) = k^2 + 2k + 1 = (k + 1)^2. By induction, the claim holds for all positive integers n. QED.

## Direct check
Table 1 compares the sum with n^2 for small n.

| n | S(n) | n^2 |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 4 | 4 |
| 3 | 9 | 9 |
| 4 | 16 | 16 |
| 5 | 25 | 25 |

## Conclusion
The identity holds for all positive integers, as the proof shows and the table illustrates for n up to 5.
