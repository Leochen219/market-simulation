"""Deterministic anonymous G3 schedules with a conservative two-use cap."""
from collections import Counter
from itertools import combinations
import random


def _partitions(agents: tuple[str, ...]) -> list[tuple[tuple[str, ...], ...]]:
    if not agents:
        return [()]
    first = agents[0]
    result = []
    for rest in combinations(agents[1:], 2):
        group = tuple(sorted((first, *rest)))
        remain = tuple(a for a in agents if a not in group)
        for tail in _partitions(remain):
            result.append((group, *tail))
            if len(result) > 250_000:
                raise ValueError("schedule infeasible: population has too many partitions to enumerate")
    return result


def generate_g3_schedule(agents: tuple[str, ...], rounds: int = 20, seed: int = 0):
    agents = tuple(sorted(agents))
    if len(agents) < 3 or len(agents) % 3:
        raise ValueError("G3 population must be divisible by three")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    rng = random.Random(seed)
    group_count = len(agents) // 3
    if rounds * group_count > 2 * (len(agents) * (len(agents) - 1) * (len(agents) - 2) // 6):
        raise ValueError("schedule infeasible under two-use triple limit")
    if len(agents) == 6:
        partitions = _partitions(agents)
        rng.shuffle(partitions)
        if rounds > 2 * len(partitions):
            raise ValueError("schedule infeasible under two-use triple limit")
        return tuple(partitions[(index // 2) % len(partitions)] for index in range(rounds))
    if len(agents) == 3 and rounds > 2:
        raise ValueError("schedule infeasible under two-use triple limit")
    used: Counter[tuple[str, ...]] = Counter()
    chosen: list[tuple[tuple[str, ...], ...]] = []
    # Randomized constructive partitions scale to the published n=30/39 treatments.
    # Restarting a seeded stream avoids enumerating Bell-sized partition spaces.
    for _attempt in range(100):
        used.clear()
        chosen.clear()
        for _round in range(rounds):
            for _candidate in range(1000):
                order = list(agents)
                rng.shuffle(order)
                partition = tuple(tuple(sorted(order[i:i + 3])) for i in range(0, len(order), 3))
                if all(used[group] < 2 for group in partition):
                    chosen.append(partition)
                    used.update(partition)
                    break
            else:
                break
        if len(chosen) == rounds:
            return tuple(chosen)
    raise ValueError("schedule infeasible under two-use triple limit after bounded deterministic search")
