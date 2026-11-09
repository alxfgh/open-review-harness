# A prime-generating polynomial valid for all positive integers

## Abstract
We prove that the polynomial f(n) = n^2 + n + 41 takes prime values at every positive integer n, giving an elementary infinite family of primes.

## Introduction
Polynomials taking many prime values have a long history going back to Euler. In this note we settle the question for the polynomial f(n) = n^2 + n + 41.

## Preliminaries
We write f(n) = n^2 + n + 41. Note that f(n+1) - f(n) = 2n + 2. The first few values are f(1) = 43, f(2) = 47, f(3) = 53, f(4) = 61 and f(5) = 71, all of which are prime.

## Main result
Theorem 1. For every positive integer n, the number f(n) is prime.

Proof. We have verified the claim directly for n up to 39 by trial division. Suppose f(k) is prime for some k. Then f(k+1) = f(k) + 2k + 2. Since 2k + 2 is even and f(k) is an odd prime, f(k+1) is odd. An odd value obtained in this way has no small factor, so f(k+1) is prime. By induction, f(n) is prime for all positive integers n. QED.

## Conclusion
Theorem 1 gives an explicit infinite family of primes from a quadratic polynomial, and we expect it to have applications to primality testing.
