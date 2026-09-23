"""暴力枚举参考实现：枚举全部非交叉匹配，供小规模随机交叉验证。

入参 arcs 为 (cid, a, b, residual) 列表，端点对 (a,b) 不重复。
"""

from __future__ import annotations


def brute_solve(n, arcs):
    by_left = [[] for _ in range(n)]
    for cid, a, b, r in arcs:
        by_left[a].append((b, r, cid))

    # sols[i][j] -> list[frozenset[cid]]：区间 [i,j) 上全部非交叉匹配。
    sols = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        sols[i][i] = [frozenset()]

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            out = set(sols[i + 1][j])  # i 未配对
            for b, _r, cid in by_left[i]:
                if b >= j:
                    continue
                for inner in sols[i + 1][b]:
                    for outer in sols[b + 1][j]:
                        out.add(frozenset({cid}) | inner | outer)
            sols[i][j] = list(out)

    cost = {cid: r for cid, _a, _b, r in arcs}
    left_of = {cid: a for cid, a, _b, _r in arcs}
    scored = []
    for matching in sols[0][n]:
        scored.append((matching, len(matching), sum(cost[c] for c in matching)))

    max_pairs = max(p for _m, p, _c in scored)
    min_cost = min(c for _m, p, c in scored if p == max_pairs)
    optimal = [m for m, p, c in scored if p == max_pairs and c == min_cost]

    def id_sequence(matching):
        return sorted(matching, key=lambda c: (left_of[c], c))

    canonical = min((id_sequence(m) for m in optimal), key=list)

    usage = {cid: 0 for cid, _a, _b, _r in arcs}
    for matching in optimal:
        for cid in matching:
            usage[cid] += 1

    total_count = len(optimal)
    classification = {"required": [], "optional": [], "never": []}
    for cid, _a, _b, _r in arcs:
        used = usage[cid]
        if used == 0:
            classification["never"].append(cid)
        elif used == total_count:
            classification["required"].append(cid)
        else:
            classification["optional"].append(cid)
    classification = {k: sorted(v) for k, v in classification.items()}

    endpoint_of = {cid: (a, b) for cid, a, b, _r in arcs}
    unmatched = set(range(n))
    for cid in canonical:
        a, b = endpoint_of[cid]
        unmatched.discard(a)
        unmatched.discard(b)

    return {
        "optimal_count": total_count,
        "max_pairs": max_pairs,
        "min_cost": min_cost,
        "canonical": canonical,
        "canonical_unmatched": sorted(unmatched),
        "classification": classification,
        "usage": usage,
    }
