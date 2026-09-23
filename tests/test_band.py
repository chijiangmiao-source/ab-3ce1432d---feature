"""audit_band（POST /audit-band）单元测试。

覆盖：
* 高残差替代方案进入容差带后归属变化（required/never -> optional）；
* 任意精度十进制的超额分层计数（含超大分层计数的闭式对照）；
* tolerance=0 与 /audit 完全一致（固定场景 + 随机暴力对照）；
* tolerance 越界/类型错误带字段路径且不夹带结果；
* 随机小规模输入与暴力枚举的谱、规范解、分类、出现量全面对照。
"""

from __future__ import annotations

import itertools
import random

import pytest

from solver import MAX_HITS, MAX_TOLERANCE, ValidationError, audit, audit_band

from tests.brute import brute_band


def make_hits(n, prefix="h"):
    return [{"id": f"{prefix}{k}", "position": k * 10} for k in range(n)]


def cand(cid, left, right, residual):
    return {
        "id": cid,
        "left_endpoint": left,
        "right_endpoint": right,
        "residual": residual,
    }


def band_payload(n, candidates, tolerance):
    return {"hits": make_hits(n), "candidates": candidates, "tolerance": tolerance}


# ----------------------------------------------------------- 归属变化固定场景


def test_high_residual_alternative_changes_attribution():
    # 2 对的两种完美匹配：嵌套 {out,in} 残差 0，顺序 {left,right} 残差 5。
    hits = make_hits(4)
    candidates = [
        cand("cheap_out", "h0", "h3", 0),
        cand("cheap_in", "h1", "h2", 0),
        cand("alt_left", "h0", "h1", 0),
        cand("alt_right", "h2", "h3", 5),
    ]

    zero = audit({"hits": hits, "candidates": candidates})
    assert zero["total_residual"] == 0
    assert set(zero["classification"]["required"]) == {"cheap_out", "cheap_in"}
    assert set(zero["classification"]["never"]) == {"alt_left", "alt_right"}

    # tolerance=4：替代方案残差 5 仍在带外，归属不变，带内只有一个方案。
    tight = audit_band(band_payload(4, candidates, 4))
    assert tight["min_residual"] == 0
    assert tight["band_residual_limit"] == 4
    assert tight["counts_by_excess"] == {"0": "1"}
    assert tight["total_count"] == "1"
    assert set(tight["classification"]["required"]) == {"cheap_out", "cheap_in"}
    assert set(tight["classification"]["never"]) == {"alt_left", "alt_right"}

    # tolerance=5：高残差替代进入带内，两条弧的归属全部翻转为 optional。
    wide = audit_band(band_payload(4, candidates, 5))
    assert wide["min_residual"] == 0
    assert wide["band_residual_limit"] == 5
    assert wide["counts_by_excess"] == {"0": "1", "5": "1"}
    assert wide["total_count"] == "2"
    assert wide["classification"] == {
        "required": [],
        "optional": ["alt_left", "alt_right", "cheap_in", "cheap_out"],
        "never": [],
    }
    # 规范解在整个容差带上按 id 序列裁决：'alt_left' < 'cheap_out'，
    # 取残差 5 的顺序方案；未配对击中为空。
    assert [p["id"] for p in wide["canonical_pairs"]] == ["alt_left", "alt_right"]
    assert wide["canonical_residual"] == 5
    assert wide["unmatched_hits"] == []

    # 带内残差上界随 tolerance 放宽，但 R 本身不变。
    wider = audit_band(band_payload(4, candidates, 40))
    assert wider["counts_by_excess"] == {"0": "1", "5": "1"}
    assert wider["total_count"] == "2"


def test_band_promotes_never_candidate_via_mixed_solutions():
    # 6 点：唯一 3 对最优残差 0；弧 alt 可在残差 2 的另一种 3 对方案中出现。
    candidates = [
        cand("a", "h0", "h1", 0),
        cand("b", "h2", "h3", 0),
        cand("c", "h4", "h5", 0),
        # 与 (b) 争 h2/h3 且与嵌套外弧搭配的替代结构：
        cand("wrap", "h2", "h5", 0),
        cand("mid", "h3", "h4", 2),
    ]
    res0 = audit_band(band_payload(6, candidates, 0))
    assert res0["classification"]["required"] == ["a", "b", "c"]
    assert set(res0["classification"]["never"]) == {"wrap", "mid"}

    res2 = audit_band(band_payload(6, candidates, 2))
    # 两个方案：{a,b,c} 残差 0；{a,wrap,mid} 残差 2。
    assert res2["counts_by_excess"] == {"0": "1", "2": "1"}
    assert res2["total_count"] == "2"
    assert res2["classification"]["required"] == ["a"]
    assert set(res2["classification"]["optional"]) == {"b", "c", "wrap", "mid"}
    # 规范解：[a, b, c] 与 [a, wrap, mid]，b < wrap，仍为残差 0 方案。
    assert [p["id"] for p in res2["canonical_pairs"]] == ["a", "b", "c"]
    assert res2["canonical_residual"] == 0


# ----------------------------------------------------------- 超大分层计数


def test_huge_stratified_counts_exact():
    # 每 3 个击中一个独立块，块内 3 条弧，两种 1 对选择：
    #   z{a}=(a,a+2) 残差 0；a{a}=(a,a+1) 残差 0；b{a}=(a+1,a+2) 残差 1。
    # 60 块独立 => 超额 e 的方案数 = C(60,e)·2^(60-e)，总量 3^60 量级。
    n = 180
    hits = make_hits(n)
    candidates = []
    for a in range(0, n, 3):
        candidates += [
            cand(f"z{a}", f"h{a}", f"h{a+2}", 0),
            cand(f"a{a}", f"h{a}", f"h{a+1}", 0),
            cand(f"b{a}", f"h{a+1}", f"h{a+2}", 1),
        ]

    res = audit_band(band_payload(n, candidates, MAX_TOLERANCE))
    assert res["min_residual"] == 0
    assert res["paired_hits"] == 120
    from math import comb

    for e in range(MAX_TOLERANCE + 1):
        assert res["counts_by_excess"][str(e)] == str(comb(60, e) * 2 ** (60 - e))
    # 截断：超额 41 不在响应中。
    assert "41" not in res["counts_by_excess"]
    expected_total = sum(comb(60, e) * 2 ** (60 - e) for e in range(MAX_TOLERANCE + 1))
    assert int(res["total_count"]) == expected_total
    assert int(res["optimal_count"]) == expected_total
    # 29 位十进制整数，确为任意精度而非浮点。
    assert len(res["total_count"]) >= 28
    # 规范解每块取 id 最小的 a{a}（残差 0）。
    assert [p["id"] for p in res["canonical_pairs"]] == [
        f"a{a}" for a in range(0, n, 3)
    ]
    assert res["canonical_residual"] == 0
    assert res["unmatched_hits"] == [f"h{k}" for k in range(n) if k % 3 == 2]
    # z/a 弧总在带内（可构造超额 0），b 弧只出现在部分方案：全部可选。
    assert set(res["classification"]["optional"]) == {c["id"] for c in candidates}


def test_max_scale_band_performance():
    n = MAX_HITS
    rng = random.Random(42)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = [
        cand(f"c{k:04d}", f"h{a}", f"h{b}", rng.choice([0, 1, 2, 5, 9]))
        for k, (a, b) in enumerate(possible[:4000])
    ]
    res = audit_band(band_payload(n, candidates, MAX_TOLERANCE))
    assert int(res["total_count"]) >= int(res["counts_by_excess"]["0"])
    assert len(res["canonical_pairs"]) * 2 == res["paired_hits"]


# ----------------------------------------------------------- 零容差回归


def test_zero_tolerance_matches_audit_fixed():
    hits = make_hits(6)
    candidates = [
        cand("bait", "h0", "h3", 0),
        cand("inner", "h1", "h2", 10),
        cand("tail", "h4", "h5", 1),
        cand("seq01", "h0", "h1", 1),
        cand("seq23", "h2", "h3", 1),
    ]
    a = audit({"hits": hits, "candidates": candidates})
    b = audit_band(band_payload(6, candidates, 0))
    assert b["counts_by_excess"] == {"0": a["optimal_count"]}
    assert b["total_count"] == a["optimal_count"]
    assert b["optimal_count"] == a["optimal_count"]
    assert b["min_residual"] == a["total_residual"]
    assert b["total_residual"] == a["total_residual"]
    assert b["canonical_residual"] == a["total_residual"]
    assert b["canonical_pairs"] == a["canonical_pairs"]
    assert b["unmatched_hits"] == a["unmatched_hits"]
    assert b["classification"] == a["classification"]
    assert b["paired_hits"] == a["paired_hits"]


@pytest.mark.parametrize("seed", range(40))
def test_zero_tolerance_matches_audit_random(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(4, 10)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    f"h{a}",
                    f"h{b}",
                    rng.choice([0, 0, 1, 3, 8]),
                )
            )
    a = audit({"hits": make_hits(n), "candidates": candidates})
    b = audit_band(band_payload(n, candidates, 0))
    assert b["total_count"] == a["optimal_count"]
    assert b["counts_by_excess"] == {"0": a["optimal_count"]}
    assert b["min_residual"] == a["total_residual"]
    assert b["canonical_pairs"] == a["canonical_pairs"]
    assert b["unmatched_hits"] == a["unmatched_hits"]
    assert b["classification"] == a["classification"]


# ----------------------------------------------------------- 非法 tolerance


def invalid_band(payload):
    with pytest.raises(ValidationError) as exc:
        audit_band(payload)
    return exc.value.errors


def test_invalid_tolerance_errors():
    base = {"hits": make_hits(4), "candidates": []}

    for bad in (-1, 41, 100, "5", 5.0, True, False, None, [], {}):
        errors = invalid_band({**base, "tolerance": bad})
        assert any(e["field"] == "/tolerance" for e in errors), bad

    errors = invalid_band(dict(base))
    assert any(e["field"] == "/tolerance" for e in errors)

    # 边界值合法。
    assert audit_band({**base, "tolerance": 0})["tolerance"] == 0
    assert audit_band({**base, "tolerance": 40})["tolerance"] == 40


def test_tolerance_error_carries_no_results():
    payload = band_payload(4, [cand("p", "h0", "h3", 0)], 41)
    errors = invalid_band(payload)
    assert any(e["field"] == "/tolerance" for e in errors)
    # ValidationError 只暴露 errors，不夹带任何审计字段。
    assert not hasattr(errors, "optimal_count")

    # tolerance 与其它字段错误并存时一并报告。
    bad = {"hits": make_hits(3), "candidates": [], "tolerance": "x"}
    errors = invalid_band(bad)
    fields = {e["field"] for e in errors}
    assert "/tolerance" in fields
    assert "/hits" in fields


def test_audit_route_ignores_tolerance_field():
    # 原 /audit 对额外字段保持兼容，不要求也不消费 tolerance。
    payload = band_payload(4, [cand("p", "h0", "h3", 0)], 40)
    res = audit(payload)
    assert set(res.keys()) == {
        "optimal_count",
        "paired_hits",
        "total_residual",
        "canonical_pairs",
        "unmatched_hits",
        "classification",
    }


# ----------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(80))
def test_matches_bruteforce_band(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 9)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    candidates = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.45:
            candidates.append(
                cand(
                    f"cid{next(counter):03d}",
                    f"h{a}",
                    f"h{b}",
                    rng.choice([0, 0, 1, 2, 3, 7]),
                )
            )
    tolerance = rng.choice([0, 1, 2, 3, 5, 8, 40])

    arc_records = [
        (
            c["id"],
            int(c["left_endpoint"][1:]),
            int(c["right_endpoint"][1:]),
            c["residual"],
        )
        for c in candidates
    ]
    res = audit_band(band_payload(n, candidates, tolerance))
    ref = brute_band(n, arc_records, tolerance)

    assert res["min_residual"] == ref["min_cost"]
    assert res["paired_hits"] == 2 * ref["max_pairs"]
    got_counts = {int(k): int(v) for k, v in res["counts_by_excess"].items()}
    assert got_counts == ref["counts"]
    assert int(res["total_count"]) == ref["total"]
    assert [p["id"] for p in res["canonical_pairs"]] == ref["canonical"]
    assert res["canonical_residual"] == ref["canonical_cost"]
    assert res["unmatched_hits"] == [f"h{k}" for k in ref["canonical_unmatched"]]
    assert res["classification"] == ref["classification"]

    for cid in res["classification"]["required"]:
        assert ref["usage"][cid] == ref["total"]
    for cid in res["classification"]["never"]:
        assert ref["usage"][cid] == 0
    for cid in res["classification"]["optional"]:
        assert 0 < ref["usage"][cid] < ref["total"]
    # 出现次数总和守恒：所有候选 usage 之和 = 方案数 × 每方案对数。
    assert sum(ref["usage"].values()) == ref["total"] * ref["max_pairs"]

    # 白盒：内部 _solve 的每条候选带内出现次数须与暴力枚举精确一致。
    from solver import _solve

    internal = _solve(n, arc_records, tolerance)
    assert internal["used_count"] == ref["usage"]
