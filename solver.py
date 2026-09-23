"""硅微条击中配对审计求解器。

输入校验通过后，求解点列上的带权非交叉匹配问题（允许弧段相互嵌套）：

* 击中沿位置严格递增排列，候选对连接其中两个击中（左端点位置 < 右端点位置）；
* 选中的对端点互异，按位置绘制后两两不交叉（允许嵌套与并列）；
* 目标依次为：最大化已配对击中数（等价于最大化对数 * 2）、最小化残差总和；
* 在所有达到前两级目标的方案上统计任意精度方案数，输出按左端位置顺序下
  配对标识序列字典序最小的规范方案，并把每条候选对判为 required / optional / never。

两种审计共用同一套区间分解：

* ``audit``（/audit，容差 0）：只接受残差恰好为全局最小值 R 的最优方案；
* ``audit_band``（/audit-band，容差 t ∈ [0,40]）：先求最大配对数及其最低残差 R，
  再一次性纳入配对数相同、残差不超过 R+t 的全部方案，按残差超额分组计数，
  并据整个容差带统计候选对的全局出现量。

带内求解不按残差阈值重复枚举，而是在一次 inside/outside 区间 DP 上维护
"截断残差谱"：对每个区间 [i,j) 只保留相对其局部最优残差 C[i][j] 的超额
0..t 的方案数映射。非负残差保证超预算的局部谱不可能被全局带内容许，
而配对数最优的规则在父子区间上把超额可加地分解（望远镜求和），因此
谱的截断卷积即精确合并，无重跑、无方案枚举。

所有计数使用 Python 任意精度整数；复杂度
O(n·|C| + n³ + n²·t + n·|C|·t²)（δ>t 的规则在合并前即被剪枝），
n <= 180、|C| <= 4000、t <= 40。
"""

from __future__ import annotations

from typing import Any, Optional

MIN_HITS = 4
MAX_HITS = 180
MAX_CANDIDATES = 4000
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
    require_tolerance: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[str, int, int, int]], Optional[int]]:
    errors: list[dict[str, str]] = []

    if not isinstance(payload, dict):
        raise ValidationError([{"field": "", "message": "请求体必须是 JSON 对象"}])

    if "hits" not in payload:
        _err(errors, "/hits", "缺少 hits 字段")
    if "candidates" not in payload:
        _err(errors, "/candidates", "缺少 candidates 字段")
    # 缺少 hits/candidates 无法继续解析，立即返回；tolerance 的问题不阻断
    # 其余字段校验，以便一次性返回全部字段路径错误。
    if any(e["field"] in ("/hits", "/candidates") for e in errors):
        raise ValidationError(errors)

    tolerance: Optional[int] = None
    # tolerance 仅属于 /audit-band；/audit 严格保持旧行为，忽略任何多余字段。
    if require_tolerance:
        if "tolerance" not in payload:
            _err(errors, "/tolerance", "缺少 tolerance 字段")
        else:
            raw_tol = payload["tolerance"]
            if not _is_int(raw_tol):
                _err(errors, "/tolerance", "tolerance 必须是 0 到 40 的整数")
            elif not 0 <= raw_tol <= MAX_TOLERANCE:
                _err(
                    errors,
                    "/tolerance",
                    f"tolerance 必须在 0 到 {MAX_TOLERANCE} 之间，收到 {raw_tol}",
                )
            else:
                tolerance = raw_tol

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

    if not errors:
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


# 谱：dict[超额 d(0..tol) -> 任意精度方案数]，只保留可达的超额。
Spectrum = dict[int, int]


def _shift_merge(target: Spectrum, source: Spectrum, shift: int, tol: int) -> None:
    """把 source 整体平移 shift 后并入 target（忽略超额超过 tol 的部分）。"""
    if shift > tol:
        return
    limit = tol - shift
    for d, w in source.items():
        if d <= limit:
            nd = d + shift
            target[nd] = target.get(nd, 0) + w


def _conv_shift_merge(
    target: Spectrum,
    left: Spectrum,
    right: Spectrum,
    shift: int,
    tol: int,
) -> None:
    """left 与 right 的截断卷积平移 shift 后并入 target。

    仅枚举超额之和不超过 tol-shift 的组合；遍历较小的谱以降低常数。
    """
    if shift > tol:
        return
    limit = tol - shift
    if len(left) > len(right):
        left, right = right, left
    for dl, wl in left.items():
        if dl > limit:
            continue
        rem = limit - dl
        for dr, wr in right.items():
            if dr <= rem:
                nd = shift + dl + dr
                target[nd] = target.get(nd, 0) + wl * wr


def _conv_prefix(left: Spectrum, right: Spectrum, limit: int) -> list[int]:
    """返回前缀计数 prefix[t] = Σ_{dl+dr ≤ t} left[dl]*right[dr]，t=0..limit。"""
    conv: Spectrum = {}
    if len(left) > len(right):
        left, right = right, left
    for dl, wl in left.items():
        if dl > limit:
            continue
        rem = limit - dl
        for dr, wr in right.items():
            if dr <= rem:
                d = dl + dr
                conv[d] = conv.get(d, 0) + wl * wr
    prefix = [0] * (limit + 1)
    running = 0
    for t in range(limit + 1):
        running += conv.get(t, 0)
        prefix[t] = running
    return prefix


def _solve_band(
    hits: list[dict[str, Any]],
    candidates: list[tuple[str, int, int, int]],
    tol: int,
) -> dict[str, Any]:
    """区间分解上的截断残差谱 inside/outside DP。

    返回 P/C/谱/外部谱与候选在带内的出现次数，供两种审计组装响应。
    """

    n = len(hits)

    # arcs[i]: 以位置 i 为左端点的候选 (右端点, 残差, id)，按右端点排序。
    arcs: list[list[tuple[int, int, str]]] = [[] for _ in range(n)]
    for cid, a, b, r in candidates:
        arcs[a].append((b, r, cid))
    for lst in arcs:
        lst.sort()

    # ---------- 第一遍：最大配对数 P 与最低残差 C（全部区间） ----------
    # P[i][j]/C[i][j]：区间 [i,j) 上的最大对数及其最小残差。
    P = [[0] * (n + 1) for _ in range(n + 1)]
    C = [[0] * (n + 1) for _ in range(n + 1)]

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            best_pairs = P[i + 1][j]
            best_cost = C[i + 1][j]
            for k, r, _cid in arcs[i]:
                if k >= j:
                    break
                pairs = 1 + P[i + 1][k] + P[k + 1][j]
                cost = r + C[i + 1][k] + C[k + 1][j]
                if pairs > best_pairs or (pairs == best_pairs and cost < best_cost):
                    best_pairs = pairs
                    best_cost = cost
            P[i][j] = best_pairs
            C[i][j] = best_cost

    # ---------- 第二遍：截断残差谱 inside DP ----------
    # S[i][j][d]：区间 [i,j) 上达到 P[i][j] 对、残差为 C[i][j]+d 的方案数。
    # 关键：配对数最优的父子规则把全局超额可加地分解为
    #   δ(规则) = 规则残差 + ΣC(子区间) - C(父区间) >= 0，
    # 整棵规则树的 δ 望远镜求和恰为「总残差 - C[0][n]」，故只需保留 d<=tol。
    S: list[list[Spectrum]] = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        S[i][i] = {0: 1}

    for length in range(1, n + 1):
        for i in range(0, n - length + 1):
            j = i + length
            acc: Spectrum = {}
            # 规则 1：i 未配对（仅当保持最大对数）。
            if P[i + 1][j] == P[i][j]:
                _shift_merge(acc, S[i + 1][j], C[i + 1][j] - C[i][j], tol)
            # 规则 2：i 与 k 配对，内部 [i+1,k) 与外部 [k+1,j) 独立。
            for k, r, _cid in arcs[i]:
                if k >= j:
                    break
                if 1 + P[i + 1][k] + P[k + 1][j] != P[i][j]:
                    continue
                delta = r + C[i + 1][k] + C[k + 1][j] - C[i][j]
                if delta > tol:
                    continue
                _conv_shift_merge(acc, S[i + 1][k], S[k + 1][j], delta, tol)
            S[i][j] = acc

    root_spectrum = S[0][n]
    total_ways = sum(root_spectrum.values())
    R = C[0][n]

    # ---------- 第三遍：截断残差谱 outside DP ----------
    # O[i][j][c]：[i,j) 以配对数最优子区间出现在某根带内方案中时，
    # 区间外上下文（祖先规则 δ 与兄弟子树超额之和）恰为 c 的方案数。
    Out: list[list[Spectrum]] = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    Out[0][n] = {0: 1}

    for length in range(n, 0, -1):
        for h in range(0, n - length + 1):
            m = h + length
            outside = Out[h][m]
            if not outside:
                continue
            # 跳过规则：父 [h,m) -> 子 [h+1,m)。
            if P[h + 1][m] == P[h][m]:
                ds = C[h + 1][m] - C[h][m]
                if ds <= tol:
                    target = Out[h + 1][m]
                    limit = tol - ds
                    for e, w in outside.items():
                        if e <= limit:
                            nd = e + ds
                            target[nd] = target.get(nd, 0) + w
            # 配对规则：父 [h,m) 经弧 (h,k) -> 左子 [h+1,k)、右子 [k+1,m)。
            for k, r, _cid in arcs[h]:
                if k >= m:
                    break
                if 1 + P[h + 1][k] + P[k + 1][m] != P[h][m]:
                    continue
                delta = r + C[h + 1][k] + C[k + 1][m] - C[h][m]
                if delta > tol:
                    continue
                sl = S[h + 1][k]
                sr = S[k + 1][m]
                left_target = Out[h + 1][k]
                right_target = Out[k + 1][m]
                # 左子上下文 = 父上下文 + δ + 右兄弟超额。
                for e, w in outside.items():
                    if e + delta > tol:
                        continue
                    rem = tol - e - delta
                    for dr, wr in sr.items():
                        if dr <= rem:
                            nd = e + delta + dr
                            left_target[nd] = left_target.get(nd, 0) + w * wr
                # 右子上下文 = 父上下文 + δ + 左兄弟超额。
                for e, w in outside.items():
                    if e + delta > tol:
                        continue
                    rem = tol - e - delta
                    for dl, wl in sl.items():
                        if dl <= rem:
                            nd = e + delta + dl
                            right_target[nd] = right_target.get(nd, 0) + w * wl

    # ---------- 候选对的带内全局出现量 ----------
    # 弧 (a,b) 只能作为区间 [a,m) 的首步配对规则出现（m>b）：
    # 出现量 = Σ_m Σ_{c+δ+dl+dr ≤ tol} O[a][m][c]·S[a+1][b][dl]·S[b+1][m][dr]。
    used_count: dict[str, int] = {}
    for cid, a, b, r in candidates:
        interior = S[a + 1][b]
        count = 0
        for m in range(b + 1, n + 1):
            if 1 + P[a + 1][b] + P[b + 1][m] != P[a][m]:
                continue
            delta = r + C[a + 1][b] + C[b + 1][m] - C[a][m]
            if delta > tol:
                continue
            prefix = _conv_prefix(interior, S[b + 1][m], tol - delta)
            for c, wc in Out[a][m].items():
                limit = tol - delta - c
                if limit >= 0:
                    count += wc * prefix[limit]
        used_count[cid] = count

    # ---------- 带内规范解（按左端位置顺序的 id 序列字典序最小） ----------
    # 在预算 b（相对 C[i][j] 的超额上界）内贪心地取首标识最小的可行规则：
    # 配对规则首标识即其 cid；跳过规则的首标识是子区间规范序列的首个标识，
    # 空序列（该子区间无对）最小。可行规则只需 δ<=b：子谱在超额 0 处恒非空。
    # decision 记录首步与子预算划分，供一次性遍历收集配对与未配对位置：
    #   ("empty",)                         区间内无配对；
    #   ("s", ds, b-ds)                    跳过 i；
    #   ("p", cid, k, δ, e_left, e_right)  以弧 (i,k) 配对。
    Plan = tuple[tuple[str, ...], tuple[Any, ...]]
    plan_memo: dict[tuple[int, int, int], Plan] = {}

    def plan(i: int, j: int, b: int) -> Plan:
        key = (i, j, b)
        cached = plan_memo.get(key)
        if cached is not None:
            return cached
        if i == j:
            result: Plan = ((), ("empty",))
            plan_memo[key] = result
            return result

        # 候选首步：(首标识, 类型, ...)。首标识 None 代表空序列（最小）。
        skip_lead: Optional[Optional[str]] = None
        skip_ids: tuple[str, ...] = ()
        skip_ds = 0
        have_skip = False
        if P[i + 1][j] == P[i][j]:
            ds = C[i + 1][j] - C[i][j]
            if ds <= b:
                have_skip = True
                skip_ds = ds
                skip_ids = plan(i + 1, j, b - ds)[0]
                skip_lead = skip_ids[0] if skip_ids else None

        best_pair: Optional[tuple[str, int, int]] = None  # (cid, k, δ)
        for k, r, cid in arcs[i]:
            if k >= j:
                break
            if 1 + P[i + 1][k] + P[k + 1][j] != P[i][j]:
                continue
            delta = r + C[i + 1][k] + C[k + 1][j] - C[i][j]
            if delta <= b and (best_pair is None or cid < best_pair[0]):
                best_pair = (cid, k, delta)

        # 规则只按首标识裁决：同一起点候选 id 唯一，配对规则间不会平局。
        pair_wins = best_pair is not None and (
            not have_skip
            or (skip_lead is not None and best_pair[0] < skip_lead)
        )

        if not pair_wins:
            decision: tuple[Any, ...]
            if have_skip:
                ids = skip_ids
                decision = ("s", skip_ds, b - skip_ds)
            else:
                ids = ()
                decision = ("empty",)
        else:
            cid, k, delta = best_pair  # type: ignore[misc]
            budget = b - delta
            # 在 (e_left, e_right) 且 e_left+e_right<=budget 中最小化
            # cid + L(e_left) + R(e_right)：先取 L 的字典序最小者，
            # 在并列的 e_left 中取最小值（R 的预算随预算单调不减惠），
            # 剩余预算全部给 R。
            left_star: Optional[tuple[str, ...]] = None
            e_left = 0
            for e in range(0, budget + 1):
                seq = plan(i + 1, k, e)[0]
                if left_star is None or seq < left_star:
                    left_star = seq
                    e_left = e
            e_right = budget - e_left
            right_ids = plan(k + 1, j, e_right)[0]
            ids = (cid,) + left_star + right_ids  # type: ignore[operator]
            decision = ("p", cid, k, delta, e_left, e_right)

        result: Plan = (ids, decision)
        plan_memo[key] = result
        return result

    canonical_ids, root_decision = plan(0, n, tol)

    def canonical_excess_of(i: int, j: int, b: int) -> int:
        """按 plan 的规则树累加实际超额。"""
        ids, decision = plan(i, j, b)
        if decision[0] == "empty":
            return 0
        if decision[0] == "s":
            return decision[1] + canonical_excess_of(i + 1, j, decision[2])
        _cid, k, delta, e_left, e_right = decision[1:]
        return (
            delta
            + canonical_excess_of(i + 1, k, e_left)
            + canonical_excess_of(k + 1, j, e_right)
        )

    canonical_excess = canonical_excess_of(0, n, tol)

    # 沿同一规则树收集未配对位置。
    unmatched_idx: list[int] = []

    def collect(i: int, j: int, b: int) -> None:
        while i < j:
            decision = plan(i, j, b)[1]
            if decision[0] == "empty":
                # 区间无对：其中位置全部未配对（在父循环里逐个跳过）。
                unmatched_idx.extend(range(i, j))
                return
            if decision[0] == "s":
                unmatched_idx.append(i)
                b = decision[2]
                i += 1
            else:
                _cid, k, _delta, e_left, e_right = decision[1:]
                collect(i + 1, k, e_left)
                b = e_right
                i = k + 1

    collect(0, n, tol)

    return {
        "n": n,
        "hits": hits,
        "candidates": candidates,
        "P": P,
        "C": C,
        "root_spectrum": root_spectrum,
        "total_ways": total_ways,
        "minimum_residual": R,
        "used_count": used_count,
        "canonical_ids": list(canonical_ids),
        "canonical_excess": canonical_excess,
        "unmatched_idx": unmatched_idx,
    }


def _canonical_pair_records(
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
    total_ways: int,
) -> dict[str, list[str]]:
    required: list[str] = []
    optional: list[str] = []
    never: list[str] = []
    for cid, _a, _b, _r in candidates:
        c = used_count[cid]
        if c == 0:
            never.append(cid)
        elif c == total_ways:
            required.append(cid)
        else:
            optional.append(cid)
    return {
        "required": sorted(required),
        "optional": sorted(optional),
        "never": sorted(never),
    }


def audit(payload: Any) -> dict[str, Any]:
    """执行完整审计（容差 0），返回可直接 JSON 序列化的结果。"""

    hits, candidates, _tol = _validate(payload)
    band = _solve_band(hits, candidates, 0)

    return {
        # 以字符串承载任意精度十进制整数，避免客户端 JSON 大整数精度损失。
        "optimal_count": str(band["total_ways"]),
        "paired_hits": 2 * band["P"][0][band["n"]],
        "total_residual": band["minimum_residual"],
        "canonical_pairs": _canonical_pair_records(
            hits, candidates, band["canonical_ids"]
        ),
        "unmatched_hits": [hits[k]["id"] for k in band["unmatched_idx"]],
        "classification": _classify(
            candidates, band["used_count"], band["total_ways"]
        ),
    }


def audit_band(payload: Any) -> dict[str, Any]:
    """容差带审计：最大配对数下，纳入残差不超过 R+tolerance 的全部方案。"""

    hits, candidates, tol = _validate(payload, require_tolerance=True)
    assert tol is not None
    band = _solve_band(hits, candidates, tol)
    n = band["n"]
    R = band["minimum_residual"]
    root_spectrum = band["root_spectrum"]

    residual_bands = [
        {
            "excess": d,
            "residual": R + d,
            "count": str(root_spectrum.get(d, 0)),
        }
        for d in range(tol + 1)
    ]

    return {
        "tolerance": tol,
        "paired_hits": 2 * band["P"][0][n],
        "minimum_residual": R,
        "band_residual_limit": R + tol,
        # 按残差超额分组的任意精度十进制方案数（超额 0..tolerance 逐档给出）。
        "residual_bands": residual_bands,
        "band_count": str(band["total_ways"]),
        "canonical_pairs": _canonical_pair_records(
            hits, candidates, band["canonical_ids"]
        ),
        "canonical_residual": R + band["canonical_excess"],
        "unmatched_hits": [hits[k]["id"] for k in band["unmatched_idx"]],
        "classification": _classify(
            candidates, band["used_count"], band["total_ways"]
        ),
    }
