"""audit_band（/audit-band 求解器）单元测试。

覆盖：非法容差校验（字段路径且不夹带结果）、零容差与 /audit 完全一致、
按超额分组的谱计数、高残差替代改变必选/可选归属、任意精度超大分层计数，
以及随机输入上与暴力容差带枚举的全面对照。
"""

from __future__ import annotations

import itertools
import random

import pytest

from solver import (
    MAX_CANDIDATES,
    MAX_HITS,
    MAX_TOLERANCE,
    MIN_HITS,
    ValidationError,
    _solve_band,
    audit,
    audit_band,
)

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


def to_arc_records(candidates, hit_ids):
    pos = {h: k for k, h in enumerate(hit_ids)}
    return [
        (c["id"], pos[c["left_endpoint"]], pos[c["right_endpoint"]], c["residual"])
        for c in candidates
    ]


# ---------------------------------------------------------------- 校验


def invalid_band(payload):
    with pytest.raises(ValidationError) as exc:
        audit_band(payload)
    return exc.value.errors


def test_tolerance_missing():
    errors = invalid_band({"hits": make_hits(4), "candidates": []})
    assert any(e["field"] == "/tolerance" for e in errors)


@pytest.mark.parametrize("bad", [41, -1, 100, 1.5, "3", True, None, [], {}])
def test_tolerance_out_of_range_or_wrong_type(bad):
    errors = invalid_band(
        {"hits": make_hits(4), "candidates": [], "tolerance": bad}
    )
    assert any(e["field"] == "/tolerance" for e in errors)


def test_tolerance_boundary_values_ok():
    for t in (0, MAX_TOLERANCE):
        res = audit_band(
            {"hits": make_hits(4), "candidates": [], "tolerance": t}
        )
        assert res["band_count"] == "1"


def test_error_response_carries_no_results():
    errors = invalid_band(
        {
            "hits": make_hits(4),
            "candidates": [],
            "tolerance": "40",
        }
    )
    assert any(e["field"] == "/tolerance" for e in errors)
    # ValidationError 本身不携带任何审计字段。
    assert set(errors[0].keys()) == {"field", "message"}

    # 容差非法时即使其余字段也有问题，仍只返回 errors。
    errors = invalid_band(
        {"hits": make_hits(3), "candidates": [], "tolerance": -1}
    )
    fields = {e["field"] for e in errors}
    assert "/tolerance" in fields
    assert "/hits" in fields


def test_hits_and_candidates_still_validated():
    errors = invalid_band(
        {"hits": make_hits(3), "candidates": [], "tolerance": 0}
    )
    assert any(e["field"] == "/hits" for e in errors)


def test_audit_ignores_tolerance_field_and_stays_compatible():
    # 旧路由必须保持请求/响应兼容：多传 tolerance 也不触发任何校验。
    payload = {"hits": make_hits(4), "candidates": [], "tolerance": "not-an-int"}
    res = audit(payload)
    assert res["optimal_count"] == "1"
    assert "band_count" not in res


# ---------------------------------------------------------------- 零容差回归


def test_zero_tolerance_identical_to_audit():
    rng = random.Random(123)
    for n in (4, 5, 8):
        ids = [f"h{k}" for k in range(n)]
        possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
        rng.shuffle(possible)
        cands = [
            cand(f"c{i:03d}", f"h{a}", f"h{b}", rng.choice([0, 1, 2, 5]))
            for i, (a, b) in enumerate(p for p in possible if rng.random() < 0.5)
        ]
        payload = {"hits": make_hits(n), "candidates": cands}
        a = audit(payload)
        b = audit_band({**payload, "tolerance": 0})

        assert b["minimum_residual"] == a["total_residual"]
        assert b["band_count"] == a["optimal_count"]
        assert b["paired_hits"] == a["paired_hits"]
        assert b["canonical_pairs"] == a["canonical_pairs"]
        assert b["canonical_residual"] == a["total_residual"]
        assert b["unmatched_hits"] == a["unmatched_hits"]
        assert b["classification"] == a["classification"]
        assert b["band_residual_limit"] == a["total_residual"]
        assert b["residual_bands"] == [
            {"excess": 0, "residual": a["total_residual"], "count": a["optimal_count"]}
        ]


# ---------------------------------------------------------------- 谱分组与归属变化


def test_residual_band_spectrum_grouping():
    # 3 个相互独立的二选一块：低残差选项 r=0，高残差替代 r=2。
    # 最大配对数每块恰 1 对；选高价选项的个数 k 决定超额 2k。
    cands = []
    n_blocks = 3
    for block in range(n_blocks):
        x = 3 * block
        cands.append(cand(f"lo{block}", f"h{x}", f"h{x+1}", 0))
        cands.append(cand(f"hi{block}", f"h{x}", f"h{x+2}", 2))
    payload = {"hits": make_hits(3 * n_blocks), "candidates": cands}

    res0 = audit_band({**payload, "tolerance": 0})
    assert res0["band_count"] == "1"
    assert res0["residual_bands"][0]["count"] == "1"
    assert set(res0["classification"]["required"]) == {"lo0", "lo1", "lo2"}
    assert set(res0["classification"]["never"]) == {"hi0", "hi1", "hi2"}

    res = audit_band({**payload, "tolerance": 4})
    assert res["minimum_residual"] == 0
    assert res["band_residual_limit"] == 4
    counts = {row["excess"]: row["count"] for row in res["residual_bands"]}
    # 超额 0:1（全低）, 2:3（一块高）, 4:3（两块高）；超额 1,3 不可达。
    assert counts[0] == "1"
    assert counts[1] == "0"
    assert counts[2] == "3"
    assert counts[3] == "0"
    assert counts[4] == "3"
    assert res["band_count"] == "7"
    # 容差带内每条候选都在某些（但非全部）方案中出现：required 集合清空。
    assert res["classification"]["required"] == []
    assert set(res["classification"]["optional"]) == {c["id"] for c in cands}
    assert res["classification"]["never"] == []


def test_high_residual_alternative_changes_attribution():
    # 零容差下并列对 low+rest 必选、嵌套高残差替代 z_alt+alt_rest 从不；
    # 放宽到 5 后替代方案入带，四条候选全部变为可选。
    payload = {
        "hits": make_hits(4),
        "candidates": [
            cand("low", "h0", "h1", 0),
            cand("rest", "h2", "h3", 0),
            cand("z_alt", "h0", "h3", 5),
            cand("alt_rest", "h1", "h2", 0),
        ],
    }
    res0 = audit_band({**payload, "tolerance": 0})
    # 最优 2 对：low+rest（残差 0）。
    assert res0["minimum_residual"] == 0
    assert set(res0["classification"]["required"]) == {"low", "rest"}
    assert set(res0["classification"]["never"]) == {"z_alt", "alt_rest"}

    res = audit_band({**payload, "tolerance": 5})
    # 带内两个方案：low+rest（超额 0）与 z_alt+alt_rest（超额 5，嵌套）。
    counts = {row["excess"]: row["count"] for row in res["residual_bands"]}
    assert counts[0] == "1"
    assert counts[5] == "1"
    assert res["band_count"] == "2"
    assert res["classification"]["required"] == []
    assert set(res["classification"]["optional"]) == {
        "low",
        "rest",
        "z_alt",
        "alt_rest",
    }
    # 规范解仍取超额 0 且首标识更小的方案。
    assert [p["id"] for p in res["canonical_pairs"]] == ["low", "rest"]
    assert res["canonical_residual"] == 0
    assert res["unmatched_hits"] == []

    # 容差 4 时高价替代仍在带外，归属与零容差一致。
    res4 = audit_band({**payload, "tolerance": 4})
    assert res4["band_count"] == "1"
    assert set(res4["classification"]["required"]) == {"low", "rest"}
    assert set(res4["classification"]["never"]) == {"z_alt", "alt_rest"}


def test_canonical_can_pick_higher_residual_when_id_smaller():
    # 高残差方案首标识字典序更小时，规范解来自非零超额层。
    payload = {
        "hits": make_hits(4),
        "candidates": [
            cand("zzz_low", "h0", "h1", 0),
            cand("tail", "h2", "h3", 0),
            cand("aaa_hi", "h0", "h3", 3),
            cand("mid", "h1", "h2", 0),
        ],
    }
    res = audit_band({**payload, "tolerance": 3})
    assert res["minimum_residual"] == 0
    assert [p["id"] for p in res["canonical_pairs"]] == ["aaa_hi", "mid"]
    assert res["canonical_residual"] == 3
    assert res["unmatched_hits"] == []
    assert res["band_count"] == "2"


# ---------------------------------------------------------------- 任意精度大计数


def test_huge_stratified_counts_exact():
    # 30 个独立二选一块（低 r=0 / 高 r=1），tol=40 容纳全部方案：
    # 总方案 2^30，超额 k 的方案数 C(30,k)，全部精确整数。
    n_blocks = 30
    cands = []
    for block in range(n_blocks):
        x = 3 * block
        cands.append(cand(f"lo{block}", f"h{x}", f"h{x+1}", 0))
        cands.append(cand(f"hi{block}", f"h{x}", f"h{x+2}", 1))
    payload = {"hits": make_hits(3 * n_blocks), "candidates": cands}
    res = audit_band({**payload, "tolerance": 40})

    assert res["minimum_residual"] == 0
    assert res["band_count"] == str(2**n_blocks)
    for row in res["residual_bands"]:
        from math import comb

        assert row["count"] == str(comb(n_blocks, row["excess"]))
    # 每条候选恰在一半方案中出现 => 可选。
    assert res["classification"]["required"] == []
    assert res["classification"]["never"] == []
    assert len(res["classification"]["optional"]) == 2 * n_blocks


def test_giant_count_exceeds_json_safe_integer():
    # 60 个三选一块（组内三条 0 残差候选），计数 3^60 远超 2^53，
    # 必须以十进制字符串精确返回。
    n_blocks = 60
    cands = []
    for block in range(n_blocks):
        x = 3 * block
        cands.append(cand(f"a{block}", f"h{x}", f"h{x+2}", 0))
        cands.append(cand(f"b{block}", f"h{x}", f"h{x+1}", 0))
        cands.append(cand(f"c{block}", f"h{x+1}", f"h{x+2}", 0))
    payload = {"hits": make_hits(3 * n_blocks), "candidates": cands}
    res = audit_band({**payload, "tolerance": 0})
    assert res["band_count"] == str(3**n_blocks)
    assert int(res["band_count"]) == 3**n_blocks
    assert res["residual_bands"][0]["count"] == str(3**n_blocks)


# ---------------------------------------------------------------- 随机暴力对照


@pytest.mark.parametrize("seed", range(80))
def test_matches_bruteforce_band(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(MIN_HITS, 9)
    tol = rng.randint(0, 6)
    ids = [f"h{k}" for k in range(n)]

    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    cands = []
    counter = itertools.count()
    for a, b in possible:
        if rng.random() < 0.4:
            cands.append(
                cand(
                    f"cid{next(counter):03d}",
                    ids[a],
                    ids[b],
                    rng.choice([0, 0, 1, 2, 3, 5]),
                )
            )

    payload = {"hits": make_hits(n), "candidates": cands}
    res = audit_band({**payload, "tolerance": tol})
    ref = brute_band(n, to_arc_records(cands, ids), tol)

    assert res["minimum_residual"] == ref["min_cost"]
    assert res["paired_hits"] == 2 * ref["max_pairs"]
    assert int(res["band_count"]) == ref["band_count"]
    for row in res["residual_bands"]:
        assert int(row["count"]) == ref["bands"][row["excess"]]
        assert row["residual"] == ref["min_cost"] + row["excess"]

    assert [p["id"] for p in res["canonical_pairs"]] == ref["canonical"]
    assert res["canonical_residual"] == ref["canonical_residual"]
    hit_ids = [f"h{k}" for k in range(n)]
    assert res["unmatched_hits"] == [hit_ids[k] for k in ref["canonical_unmatched"]]
    assert res["classification"] == ref["classification"]

    # 候选在带内全局出现量的精确值（inside-outside 截断谱的核心保证）。
    band = _solve_band(make_hits(n), to_arc_records(cands, ids), tol)
    for cid, _a, _b, _r in to_arc_records(cands, ids):
        assert band["used_count"][cid] == ref["usage"][cid]
    assert band["total_ways"] == ref["band_count"]


# ---------------------------------------------------------------- 性能


def test_max_scale_performance_band():
    n = MAX_HITS
    rng = random.Random(7)
    possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    rng.shuffle(possible)
    cands = [
        cand(f"c{k:04d}", f"h{a}", f"h{b}", rng.randrange(100))
        for k, (a, b) in enumerate(possible[:MAX_CANDIDATES])
    ]
    res = audit_band(
        {"hits": make_hits(n), "candidates": cands, "tolerance": MAX_TOLERANCE}
    )
    assert int(res["band_count"]) >= 1
    assert len(res["residual_bands"]) == MAX_TOLERANCE + 1
    assert len(res["canonical_pairs"]) * 2 == res["paired_hits"]
