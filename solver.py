"""硅微条击中配对审计求解器。

输入校验通过后，求解点列上的带权非交叉匹配问题（允许弧段相互嵌套）：

* 击中沿位置严格递增排列，候选对连接其中两个击中（左端点位置 < 右端点位置）；
* 选中的对端点互异，按位置绘制后两两不交叉（允许嵌套与并列）；
* 目标依次为：最大化已配对击中数（等价于最大化对数 * 2）、最小化残差总和；
* 在所有达到前两级目标的方案上统计任意精度方案数，输出按左端位置顺序下
  配对标识序列字典序最小的规范方案，并把每条候选对判为 required / optional / never。

两种审计共用同一套区间分解：

* ``audit``（``POST /audit``）只看唯一最低残差，等价于容差带宽度 0；
* ``audit_band``（``POST /audit-band``）先求最大配对数下的最低残差 R，再纳入
  配对数相同且残差不超过 R+tolerance 的全部方案（0 <= tolerance <= 40）。

带形审计不在每个残差阈值上重跑、也不枚举方案，而是对每个区间 ``[i,j)``
维护一条相对该区间最小残差的“截断超额谱” ``S[i][j][e]``：达到最大配对数、
残差恰为 ``R[i][j] + e`` 的方案数，仅保留 0 <= e <= tolerance。子方案的
超额非负，全局超额为各段超额之和，故任何能出现在带内全局方案中的子方案
其超额必不超过 tolerance，按 tolerance 截断无损。谱的合并是带移位的截断
计数卷积；outside 谱同理由父区间向两个子区间下发，候选弧的带内出现量在
下发配对规则时借前缀和一次求出。

所有计数使用 Python 任意精度整数；标量区间 DP 复杂度 O(n*|C| + n^3)，
谱 pass 与区间-弧 incidences 成比例、每次卷积至多 O(tolerance^2)，
tolerance <= 40、n <= 180、|C| <= 4000。
"""

from __future__ import annotations

from typing import Any, Optional

MIN_HITS = 4
MAX_HITS = 180
MAX_CANDIDATES = 4000
MIN_TOLERANCE = 0
MAX_TOLERANCE = 40


class ValidationError(Exception):
    """携带字段路径的请求校验错误。"""

    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("; ".join(e["message"] for e in errors))
        self.errors = errors


def _err(errors: list[dict[str, str]], field: str, message: str) -> None:
    errors.append({"field": field, "message": message})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(
    payload: Any,
    *,
    expect_tolerance: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]], Optional[int]]:
    errors: list[dict[str, str]] = []

    if not isinstance(payload, dict):
        raise ValidationError([{"field": "", "message": "请求体必须是 JSON 对象"}])

    if "hits" not in payload:
        _err(errors, "/hits", "缺少 hits 字段")
    if "candidates" not in payload:
        _err(errors, "/candidates", "缺少 candidates 字段")
    if expect_tolerance and "tolerance" not in payload:
        _err(errors, "/tolerance", "缺少 tolerance 字段")
    if errors:
        raise ValidationError(errors)

    raw_hits = payload["hits"]
    raw_candidates = payload["candidates"]

    if not isinstance(raw_hits, list):
        _err(errors, "/hits", "hits 必须是数组")
        raw_hits = []
    if not isinstance(raw_candidates, list):
        _err(errors, "/candidates", "candidates 必须是数组")
        raw_candidates = []

    if isinstance(raw_hits, list) and not (MIN_HITS <= len(raw_hits) <= MAX_HITS):
        _err(
            errors,
            "/hits",
            f"击中数量必须在 {MIN_HITS} 到 {MAX_HITS} 之间，收到 {len(raw_hits)}",
        )
    if isinstance(raw_candidates, list) and len(raw_candidates) > MAX_CANDIDATES:
        _err(
            errors,
            "/candidates",
            f"候选配对数量不能超过 {MAX_CANDIDATES}，收到 {len(raw_candidates)}",
        )

    # tolerance 的类型/区间错误与其余字段错误一并报告，但绝不夹带任何审计结果。
    tolerance: Optional[int] = None
    if expect_tolerance:
        raw_tolerance = payload["tolerance"]
        if not _is_int(raw_tolerance):
            _err(errors, "/tolerance", "tolerance 必须是整数")
        elif not (MIN_TOLERANCE <= raw_tolerance <= MAX_TOLERANCE):
            _err(
                errors,
                "/tolerance",
                f"tolerance 必须在 {MIN_TOLERANCE} 到 {MAX_TOLERANCE} 之间，"
                f"收到 {raw_tolerance}",
            )
        else:
            tolerance = raw_tolerance

    hits: list[dict[str, Any]] = []
    seen_hit_ids: set[str] = set()

    for k, hit in enumerate(raw_hits if isinstance(raw_hits, list) else []):
        base = f"/hits/{k}"
        if not isinstance(hit, dict):
            _err(errors, base, "击中必须是对象")
            continue
        hid = hit.get("id")
        pos = hit.get("position")
        if not isinstance(hid, str) or not hid:
            _err(errors, f"{base}/id", "击中 id 必须是非空字符串")
        elif hid in seen_hit_ids:
            _err(errors, f"{base}/id", f"击中 id 重复: {hid}")
        else:
            seen_hit_ids.add(hid)
        if not _is_int(pos):
            _err(errors, f"{base}/position", "position 必须是整数")
            pos = None
        hits.append({"id": hid, "position": pos})

    if not any(e["field"].startswith("/hits") for e in errors):
        for k in range(1, len(hits)):
            if hits[k]["position"] <= hits[k - 1]["position"]:
                _err(
                    errors,
                    f"/hits/{k}/position",
                    f"位置必须严格递增: {hits[k - 1]['position']} 之后出现 "
                    f"{hits[k]['position']}",
                )
                break

    candidate_records: list[tuple[str, int, int, int]] = []
    if not any(e["field"].startswith("/hits") for e in errors):
        index_by_id = {h["id"]: k for k, h in enumerate(hits)}
        seen_pair_ids: set[str] = set()
        seen_endpoint_pairs: set[tuple[int, int]] = set()

        for k, cand in enumerate(raw_candidates if isinstance(raw_candidates, list) else []):
            base = f"/candidates/{k}"
            if not isinstance(cand, dict):
                _err(errors, base, "候选配对必须是对象")
                continue
            cid = cand.get("id")
            left = cand.get("left_endpoint")
            right = cand.get("right_endpoint")
            residual = cand.get("residual")

            if not isinstance(cid, str) or not cid:
                _err(errors, f"{base}/id", "候选 id 必须是非空字符串")
            elif cid in seen_pair_ids:
                _err(errors, f"{base}/id", f"候选 id 重复: {cid}")
            else:
                seen_pair_ids.add(cid)

            if not _is_int(residual):
                _err(errors, f"{base}/residual", "residual 必须是非负整数")
            elif residual < 0:
                _err(errors, f"{base}/residual", f"residual 不能为负，收到 {residual}")

            if not isinstance(left, str):
                _err(errors, f"{base}/left_endpoint", "left_endpoint 必须是字符串标识")
            elif left not in index_by_id:
                _err(errors, f"{base}/left_endpoint", f"未知端点标识: {left}")

            if not isinstance(right, str):
                _err(errors, f"{base}/right_endpoint", "right_endpoint 必须是字符串标识")
            elif right not in index_by_id:
                _err(errors, f"{base}/right_endpoint", f"未知端点标识: {right}")

            if (
                isinstance(left, str)
                and isinstance(right, str)
                and left in index_by_id
                and right in index_by_id
            ):
                a = index_by_id[left]
                b = index_by_id[right]
                if a >= b:
                    _err(
                        errors,
                        f"{base}/right_endpoint",
                        "右端点位置必须严格大于左端点位置，且两端点必须不同",
                    )
                elif (a, b) in seen_endpoint_pairs:
                    _err(errors, base, f"重复端点对: ({left}, {right})")
                else:
                    seen_endpoint_pairs.add((a, b))
                    if isinstance(cid, str) and cid and _is_int(residual) and residual >= 0:
                        candidate_records.append((cid, a, b, residual))

    if errors:
        raise ValidationError(errors)

    return hits, candidate_records, tolerance


# 规则首步的紧凑表示：('s',) 跳过最左点；('p', k, cid) 以弧 (i,k) 配对。
Rule = tuple[Any, ...]


def _solve(n: int, candidates: list[tuple[str, int, int, int]], tolerance: int) -> dict[str, Any]:
    """区间分解核心。tolerance=0 时即原唯一最低残差审计。"""

    T = tolerance

    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 残差, id)。
    arcs: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
    arc_residual: dict[tuple[int, int], int] = {}
    for cid, a, b, r in candidates:
        arcs[a].append((b, r, cid))
        arc_residual[(a, b)] = r

    # ---------- 标量 inside pass：最大对数 P 与对应最低残差 R ----------
    P = [[0] * (n + 1) for _ in range(n + 1)]
    R = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            best_pairs = P[i + 1][j]
            best_cost = R[i + 1][j]
            for k, r, _cid in arcs[i]:
                if k >= j:
                    continue
                pairs = 1 + P[i + 1][k] + P[k + 1][j]
                cost = r + R[i + 1][k] + R[k + 1][j]
                if pairs > best_pairs or (pairs == best_pairs and cost < best_cost):
                    best_pairs = pairs
                    best_cost = cost
            P[i][j] = best_pairs
            R[i][j] = best_cost

    root_R = R[0][n]

    # 每个区间的可达最大配对规则及其相对最低残差的固定移位 delta >= 0：
    # 采用该规则且两个子区间都取自身最低残差时，相对本区间最低残差多付 delta。
    # delta > T 的规则不可能出现在带内，直接剔除。
    # pair_rules[i][j] 为 (delta, cid, k)，按 cid 升序（cid 全局唯一，配对
    # 候选序列首元素即 cid，故规范解只需其中 cid 最小者与 skip 序列比较）。
    pair_rules: list[list[list[tuple[int, str, int]]]] = [
        [[] for _ in range(n + 1)] for _ in range(n + 1)
    ]
    # skip_delta[i][j]：skip 可达最大对数时为 R[i+1][j]-R[i][j]，否则 None。
    skip_delta: list[list[Optional[int]]] = [
        [None] * (n + 1) for _ in range(n + 1)
    ]
    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            if P[i + 1][j] == P[i][j]:
                skip_delta[i][j] = R[i + 1][j] - R[i][j]
            rules: list[tuple[int, str, int]] = []
            for k, r, cid in arcs[i]:
                if k >= j:
                    continue
                if 1 + P[i + 1][k] + P[k + 1][j] != P[i][j]:
                    continue
                delta = r + R[i + 1][k] + R[k + 1][j] - R[i][j]
                if delta <= T:
                    rules.append((delta, cid, k))
            rules.sort(key=lambda item: item[1])
            pair_rules[i][j] = rules

    def nonzero_indices(spec: list[int]) -> list[int]:
        return [e for e, v in enumerate(spec) if v]

    def conv(
        xs: list[int],
        ys: list[int],
        limit: int = -1,
    ) -> list[int]:
        """截断计数卷积：res[q] = Σ_{x+y=q} xs[x]*ys[y]，q <= limit（默认 T）。

        仅遍历非零下标；按更稀疏的一侧外层遍历并提前截断。
        """

        if limit < 0:
            limit = T
        res = [0] * (limit + 1)
        xi = nonzero_indices(xs)
        yi = nonzero_indices(ys)
        if not xi or not yi:
            return res
        if len(xi) > len(yi):
            xi, yi = yi, xi
            xs, ys = ys, xs
        for x in xi:
            bound = limit - x
            if bound < 0:
                break
            vx = xs[x]
            for y in yi:
                if y > bound:
                    break
                res[x + y] += vx * ys[y]
        return res

    # ---------- 谱 inside pass：截断超额谱 S ----------
    # S[i][j][e]：区间 [i,j) 达到 P[i][j] 对、残差恰为 R[i][j]+e 的方案数。
    S: list[list[list[int]]] = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    # seq[i][j][s]：超额预算不超过 s 的全部方案中，按左端位置顺序的配对 id
    # 序列字典序最小者；ex[i][j][s] 为其实际超额。规范解回溯只走 O(n) 个
    # 区间节点，首步规则按相同判据即时重算，不再常驻 choice 表。
    seq: list[list[list[Optional[list[str]]]]] = [
        [[] for _ in range(n + 1)] for _ in range(n + 1)
    ]
    excess: list[list[list[int]]] = [[[] for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        S[i][i] = [1] + [0] * T
        seq[i][i] = [[] for _ in range(T + 1)]
        excess[i][i] = [0] * (T + 1)

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            ds = skip_delta[i][j]
            rules = pair_rules[i][j]

            spec = [0] * (T + 1)
            if ds is not None:
                child = S[i + 1][j]
                for e in range(T - ds + 1):
                    v = child[e]
                    if v:
                        spec[e + ds] += v
            for delta, _cid, k in rules:
                merged = conv(S[i + 1][k], S[k + 1][j], T - delta)
                for q, v in enumerate(merged):
                    if v:
                        spec[q + delta] += v
            S[i][j] = spec

            # 每个预算 s 下 cid 最小的可行配对（delta <= s）。rules 按 cid
            # 升序；cid 更小的规则已占据的预算无需再让后续规则覆盖，只需填补
            # [delta, 此前最小 delta) 这段它尚不可行的预算，整体 O(len(rules)+T)。
            arc_for_budget: list[Optional[tuple[int, str, int]]] = [None] * (T + 1)
            earlier_min_delta = T + 1
            for delta, cid, k in rules:
                if delta < earlier_min_delta:
                    for s in range(delta, earlier_min_delta):
                        arc_for_budget[s] = (delta, cid, k)
                    earlier_min_delta = delta

            seqs: list[Optional[list[str]]] = [None] * (T + 1)
            exs: list[int] = [0] * (T + 1)
            prev_seq: Optional[list[str]] = None
            for s in range(T + 1):
                best_seq: Optional[list[str]] = None
                best_ex = 0

                if ds is not None and s >= ds:
                    child_s = s - ds
                    best_seq = seq[i + 1][j][child_s]
                    best_ex = ds + excess[i + 1][j][child_s]

                rule_pick = arc_for_budget[s]
                if rule_pick is not None:
                    delta, cid, k = rule_pick
                    inner_s = s - delta
                    inner_seq = seq[i + 1][k][inner_s]
                    ei = excess[i + 1][k][inner_s]
                    # 内部先用满预算取字典序最小序列，再把剩余预算留给外部，
                    # 保证 [cid]+内部+外部 整体字典序最小。
                    outer_s = inner_s - ei
                    outer_seq = seq[k + 1][j][outer_s]
                    cand_seq = [cid, *inner_seq, *outer_seq]
                    cand_ex = delta + ei + excess[k + 1][j][outer_s]
                    if best_seq is None or cand_seq < best_seq:
                        best_seq = cand_seq
                        best_ex = cand_ex

                # 相邻预算的最小序列经常完全相同；复用同一对象，避免
                # 在 O(区间数 * T) 个槽位上各存一份 ~对数长度的列表。
                if best_seq is not None and best_seq == prev_seq:
                    best_seq = prev_seq
                prev_seq = best_seq
                seqs[s] = best_seq
                exs[s] = best_ex

            seq[i][j] = seqs
            excess[i][j] = exs

    root_spec = S[0][n]
    total = sum(root_spec)
    root_excess = excess[0][n][T]

    # ---------- 谱 outside pass：上下文谱与候选弧带内出现量 ----------
    # O[i][j][f]：[i,j) 的外部上下文（祖先弧 + 兄弟子树）相对
    # “全局最低 - 区间最低”的超额恰为 f 的方案数；与超额 e 的子方案组合后
    # 全局超额为 e+f，带内要求 e+f <= T。
    Out: list[list[list[int]]] = [
        [[0] * (T + 1) for _ in range(n + 1)] for _ in range(n + 1)
    ]
    Out[0][n][0] = 1

    used_count: dict[str, int] = {cid: 0 for cid, _a, _b, _r in candidates}

    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            onz = nonzero_indices(Out[h][m])
            if not onz:
                continue

            # skip 规则：父 [h,m) -> 子 [h+1,m)，无弧，整体平移 skip_delta。
            ds = skip_delta[h][m]
            if ds is not None:
                child = Out[h + 1][m]
                parent = Out[h][m]
                for f in onz:
                    if f + ds <= T:
                        child[f + ds] += parent[f]

            # 配对规则：父经弧 (h,k) 连内子 [h+1,k)、外子 [k+1,m)。
            parent_spec = Out[h][m]
            for delta, cid, k in pair_rules[h][m]:
                inner_spec = S[h + 1][k]
                inner_idx = nonzero_indices(inner_spec)
                outer_spec = S[k + 1][m]

                # g[q] = Σ_{f+eo=q} 上下文(f) * 外子树(eo)：内子树将收到的
                # 新上下文谱（计入弧前），同时用于本弧出现量计数。
                g = conv(parent_spec, outer_spec)
                appear = 0
                if inner_idx:
                    # 维护 g 的前缀；内超额 ei 取 q <= T-delta-ei。
                    prefix = 0
                    prefix_at: list[int] = [0] * (T + 1)
                    for q in range(T + 1):
                        prefix += g[q]
                        prefix_at[q] = prefix
                    for ei in inner_idx:
                        bound = T - delta - ei
                        if bound >= 0:
                            appear += inner_spec[ei] * prefix_at[bound]
                if appear:
                    used_count[cid] += appear

                inner_out = Out[h + 1][k]
                for q, v in enumerate(g):
                    if v and q + delta <= T:
                        inner_out[q + delta] += v

                # 外子树收到的上下文谱 = 父上下文 * 内子树，再平移 delta。
                hconv = conv(parent_spec, inner_spec, T - delta)
                outer_out = Out[k + 1][m]
                for q, v in enumerate(hconv):
                    if v:
                        outer_out[q + delta] += v

    # ---------- 规范解回溯（首步规则按 inside 同判据即时重算） ----------
    canonical_ids: list[str] = []
    unmatched_idx: list[int] = []

    def best_rule(i: int, j: int, s: int) -> Rule:
        """与 inside pass 相同的判据：skip 与 cid 最小可行配对取序列较小者。"""

        ds = skip_delta[i][j]
        best: Optional[tuple[list[str], Rule]] = None
        if ds is not None and s >= ds:
            child_s = s - ds
            best = (seq[i + 1][j][child_s], ("s",))
        for delta, cid, k in pair_rules[i][j]:
            if delta > s:
                continue
            inner_s = s - delta
            inner_seq = seq[i + 1][k][inner_s]
            outer_s = inner_s - excess[i + 1][k][inner_s]
            cand = ([cid, *inner_seq, *seq[k + 1][j][outer_s]], ("p", k, cid))
            if best is None or cand[0] < best[0]:
                best = cand
            break  # pair_rules 按 cid 升序，后续配对首元素更大。
        assert best is not None
        return best[1]

    def build(i: int, j: int, s: int) -> None:
        while i < j:
            step = best_rule(i, j, s)
            if step[0] == "s":
                ds = skip_delta[i][j]
                assert ds is not None
                unmatched_idx.append(i)
                i += 1
                s -= ds
            else:
                k, cid = step[1], step[2]
                r = arc_residual[(i, k)]
                delta = r + R[i + 1][k] + R[k + 1][j] - R[i][j]
                canonical_ids.append(cid)
                inner_s = s - delta
                build(i + 1, k, inner_s)
                ei = excess[i + 1][k][inner_s]
                # 尾部 [k+1,j) 以剩余预算在本循环内继续展开。
                s = inner_s - ei
                i = k + 1

    build(0, n, T)

    return {
        "n": n,
        "P_root": P[0][n],
        "R_root": root_R,
        "spectrum": root_spec,
        "total": total,
        "canonical_ids": canonical_ids,
        "canonical_excess": root_excess,
        "unmatched_idx": unmatched_idx,
        "used_count": used_count,
    }


def _canonical_pairs(
    hits: list[dict[str, Any]],
    candidates: list[tuple[str, int, int, int]],
    canonical_ids: list[str],
) -> list[dict[str, Any]]:
    info = {cid: (a, b, r) for cid, a, b, r in candidates}
    return [
        {
            "id": cid,
            "left_endpoint": hits[info[cid][0]]["id"],
            "right_endpoint": hits[info[cid][1]]["id"],
            "residual": info[cid][2],
        }
        for cid in canonical_ids
    ]


def _classify(
    candidates: list[tuple[str, int, int, int]],
    used_count: dict[str, int],
    total: int,
) -> dict[str, list[str]]:
    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, _a, _b, _r in candidates:
        count = used_count[cid]
        if count == 0:
            never.append(cid)
        elif count == total:
            required.append(cid)
        else:
            optional.append(cid)
    return {
        "required": sorted(required),
        "optional": sorted(optional),
        "never": sorted(never),
    }


def audit(payload: Any) -> dict[str, Any]:
    """执行唯一最低残差审计，返回可直接 JSON 序列化的结果。"""

    hits, candidates, _tolerance = _validate(payload)
    result = _solve(len(hits), candidates, 0)
    canonical_pairs = _canonical_pairs(hits, candidates, result["canonical_ids"])

    return {
        # 以字符串承载任意精度十进制整数，避免客户端 JSON 大整数精度损失。
        "optimal_count": str(result["total"]),
        "paired_hits": 2 * result["P_root"],
        "total_residual": result["R_root"],
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in result["unmatched_idx"]],
        "classification": _classify(candidates, result["used_count"], result["total"]),
    }


def audit_band(payload: Any) -> dict[str, Any]:
    """执行容差带审计：最大配对数下，残差位于 [R, R+tolerance] 的全部方案。"""

    hits, candidates, tolerance = _validate(payload, expect_tolerance=True)
    assert tolerance is not None
    result = _solve(len(hits), candidates, tolerance)

    spectrum = result["spectrum"]
    counts_by_excess = {str(e): str(v) for e, v in enumerate(spectrum) if v}
    canonical_pairs = _canonical_pairs(hits, candidates, result["canonical_ids"])
    total = result["total"]
    min_residual = result["R_root"]

    return {
        "tolerance": tolerance,
        "paired_hits": 2 * result["P_root"],
        # R：最大配对数下的最低残差。
        "min_residual": min_residual,
        # 与 /audit 同名字段保持同义，便于零容差逐字段比对。
        "total_residual": min_residual,
        "band_residual_limit": min_residual + tolerance,
        # 按残差超额（相对 R）分组的任意精度十进制方案数，仅列非零档。
        "counts_by_excess": counts_by_excess,
        "optimal_count": str(total),
        "total_count": str(total),
        "canonical_residual": min_residual + result["canonical_excess"],
        "canonical_pairs": canonical_pairs,
        "unmatched_hits": [hits[k]["id"] for k in result["unmatched_idx"]],
        "classification": _classify(candidates, result["used_count"], total),
    }
