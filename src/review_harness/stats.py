"""Small, dependency-free statistics, written out so that they are easy to check."""

import math


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """95% interval for a proportion. Behaves sensibly at 0 and at small samples."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact test for paired yes/no outcomes.

    Only pairs that differ carry information: only_a is how often A succeeded and B failed.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided exact test for the table [[a, b], [c, d]]."""
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2

    def probability(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(total, col1)

    observed = probability(a)
    low, high = max(0, col1 - row2), min(row1, col1)
    # The tolerance stops rounding error from dropping tables exactly as likely as the observed one.
    p = sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed * (1 + 1e-9))
    return min(1.0, p)


def auroc(positives: list[float], negatives: list[float]) -> float:
    """Chance that a random positive scores above a random negative, ties counting half."""
    if not positives or not negatives:
        raise ValueError("both groups need at least one score")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def auroc_interval(
    area: float, n_positive: int, n_negative: int, z: float = 1.959964
) -> tuple[float, float]:
    """95% interval for an AUROC, by Hanley and McNeil (1982). Approximate at small samples."""
    q1 = area / (2 - area)
    q2 = 2 * area**2 / (1 + area)
    variance = (
        area * (1 - area) + (n_positive - 1) * (q1 - area**2) + (n_negative - 1) * (q2 - area**2)
    ) / (n_positive * n_negative)
    half = z * math.sqrt(max(0.0, variance))
    return (max(0.0, area - half), min(1.0, area + half))


def mantel_haenszel(
    tables: list[tuple[int, int, int, int]],
) -> tuple[float | None, float]:
    """Common odds ratio and two-sided CMH test, with continuity correction, over 2x2 strata.

    Each table is (a, b, c, d) for [[a, b], [c, d]]. Strata with fewer than two papers carry no
    information and are skipped. The odds ratio is None when it is undefined.
    """
    numerator = denominator = observed = expected = variance = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        numerator += a * d / n
        denominator += b * c / n
        observed += a
        expected += (a + b) * (a + c) / n
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    odds_ratio = numerator / denominator if denominator else None
    if variance == 0:
        return odds_ratio, 1.0
    statistic = max(0.0, abs(observed - expected) - 0.5) ** 2 / variance
    return odds_ratio, math.erfc(math.sqrt(statistic / 2))


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Agreement between two yes/no raters beyond chance. None if it cannot be computed."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    a_yes = sum(1 for a, _ in pairs if a) / n
    b_yes = sum(1 for _, b in pairs if b) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if expected >= 1:
        return None
    return (observed - expected) / (1 - expected)
