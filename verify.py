"""Compose verify 服务的单次复核入口。

依次执行：
1. 代码测试（pytest 单元测试，含随机暴力交叉验证）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径 + 嵌套同优、交叉低价诱饵、空候选、非法引用四类场景）。

任一步失败即以非零退出码结束，全部通过退出码 0。
"""

from __future__ import annotations

import json
import os
import py_compile
import sys
import urllib.error
import urllib.request

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
REQUIRE_FILES = os.environ.get("REQUIRE_BUILD_FILES", "").split(",")

failures: list[str] = []


def check(condition: bool, name: str, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}{(' - ' + detail) if detail and not condition else ''}")
    if not condition:
        failures.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1. 代码测试


def run_unit_tests() -> None:
    section("代码测试 (pytest)")
    import pytest

    rc = pytest.main(["-q", "tests"])
    check(rc == 0, "pytest 单元测试", f"退出码 {rc}")


# ---------------------------------------------------------------- 2. 构建检查


def run_build_checks() -> None:
    section("构建检查")
    for path in [
        "solver.py",
        "app.py",
        "verify.py",
        os.path.join("tests", "test_solver.py"),
        os.path.join("tests", "test_band.py"),
        os.path.join("tests", "brute.py"),
    ]:
        try:
            py_compile.compile(path, doraise=True)
            ok = True
        except py_compile.PyCompileError as exc:
            ok = False
            print(exc)
        check(ok, f"语法编译: {path}")

    for fname in REQUIRE_FILES:
        fname = fname.strip()
        if not fname:
            continue
        check(os.path.isfile(fname), f"镜像内文件齐备: {fname}")

    try:
        import solver  # noqa: F401
        import app  # noqa: F401

        ok = True
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(exc)
    check(ok, "模块导入 solver/app")


# ---------------------------------------------------------------- 3. API/HTTP 冒烟


def http_request(method: str, path: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        API_BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def hits(n):
    return [{"id": f"h{k}", "position": k * 10} for k in range(n)]


def cand(cid, a, b, r):
    return {
        "id": cid,
        "left_endpoint": f"h{a}",
        "right_endpoint": f"h{b}",
        "residual": r,
    }


def run_smoke() -> None:
    section(f"API/HTTP 冒烟 ({API_BASE})")

    status, body = http_request("GET", "/health")
    check(status == 200 and body.get("status") == "ready", "健康路径 /health 报告就绪",
          f"HTTP {status} {body}")

    # 场景 1：嵌套同优 —— 顺序配对与嵌套配对同为 2 对同残差。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand("a_out", 0, 3, 1),
            cand("a_in", 1, 2, 5),
            cand("b_left", 0, 1, 3),
            cand("b_right", 2, 3, 3),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "2"
        and [p["id"] for p in body["canonical_pairs"]] == ["a_out", "a_in"]
        and set(body["classification"]["optional"])
        == {"a_out", "a_in", "b_left", "b_right"}
    )
    check(ok, "嵌套同优方案并列计数与规范解", f"HTTP {status} {body}")

    # 场景 2：交叉低价诱饵 —— 0 残差的交叉弧不得胜出。
    payload = {
        "hits": hits(6),
        "candidates": [
            cand("bait", 0, 3, 0),
            cand("inner", 1, 2, 10),
            cand("tail", 4, 5, 1),
            cand("seq01", 0, 1, 1),
            cand("seq23", 2, 3, 1),
        ],
    }
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and [p["id"] for p in body["canonical_pairs"]] == ["seq01", "seq23", "tail"]
        and "bait" in body["classification"]["never"]
        and body["total_residual"] == 3
    )
    check(ok, "交叉低价诱饵不被接受", f"HTTP {status} {body}")

    # 场景 3：合法空候选 —— 唯一空方案。
    payload = {"hits": hits(4), "candidates": []}
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 200
        and body["optimal_count"] == "1"
        and body["canonical_pairs"] == []
        and len(body["unmatched_hits"]) == 4
        and body["classification"] == {"required": [], "optional": [], "never": []}
    )
    check(ok, "合法空候选返回唯一空方案", f"HTTP {status} {body}")

    # 场景 4：非法引用 —— 错误带字段路径，且不夹带任何审计字段。
    payload = {"hits": hits(4), "candidates": [cand("bad", 0, 99, 0)]}
    # 99 不在 id 中，手工构造以模拟未知端点字符串。
    payload["candidates"][0]["right_endpoint"] = "h99"
    status, body = http_request("POST", "/audit", payload)
    ok = (
        status == 400
        and isinstance(body.get("errors"), list)
        and any(e.get("field") == "/candidates/0/right_endpoint" for e in body["errors"])
        and "optimal_count" not in body
    )
    check(ok, "非法引用返回字段路径错误且无审计结果", f"HTTP {status} {body}")

    # 附加：重复端点对、位置冲突、规模越界。
    status, body = http_request(
        "POST",
        "/audit",
        {"hits": hits(4), "candidates": [cand("a", 0, 1, 0), cand("b", 0, 1, 1)]},
    )
    check(
        status == 400
        and any(e["field"] == "/candidates/1" for e in body["errors"])
        and "optimal_count" not in body,
        "重复端点对被拒绝", f"HTTP {status} {body}",
    )

    conflict = hits(4)
    conflict[2]["position"] = conflict[1]["position"]
    status, body = http_request("POST", "/audit", {"hits": conflict, "candidates": []})
    check(
        status == 400
        and any(e["field"] == "/hits/2/position" for e in body["errors"]),
        "位置冲突被拒绝", f"HTTP {status} {body}",
    )

    status, body = http_request("POST", "/audit", {"hits": hits(3), "candidates": []})
    check(
        status == 400 and any(e["field"] == "/hits" for e in body["errors"]),
        "击中规模越界被拒绝", f"HTTP {status} {body}",
    )

    # 未知路径返回 404。
    status, _ = http_request("GET", "/nope")
    check(status == 404, "未知路径返回 404", f"HTTP {status}")


def run_band_smoke() -> None:
    section(f"容差带审计冒烟 /audit-band ({API_BASE})")

    # 场景 1：高残差替代进入容差带后归属变化。
    payload = {
        "hits": hits(4),
        "candidates": [
            cand("cheap_out", 0, 3, 0),
            cand("cheap_in", 1, 2, 0),
            cand("alt_left", 0, 1, 0),
            cand("alt_right", 2, 3, 5),
        ],
    }
    status, tight = http_request("POST", "/audit-band", {**payload, "tolerance": 4})
    ok = (
        status == 200
        and tight["min_residual"] == 0
        and tight["counts_by_excess"] == {"0": "1"}
        and set(tight["classification"]["required"]) == {"cheap_out", "cheap_in"}
        and set(tight["classification"]["never"]) == {"alt_left", "alt_right"}
    )
    check(ok, "窄容差带归属与唯一最低残差一致", f"HTTP {status} {tight}")

    status, wide = http_request("POST", "/audit-band", {**payload, "tolerance": 5})
    ok = (
        status == 200
        and wide["counts_by_excess"] == {"0": "1", "5": "1"}
        and wide["total_count"] == "2"
        and wide["band_residual_limit"] == 5
        and wide["classification"]["required"] == []
        and wide["classification"]["never"] == []
        and set(wide["classification"]["optional"])
        == {"cheap_out", "cheap_in", "alt_left", "alt_right"}
        and [p["id"] for p in wide["canonical_pairs"]] == ["alt_left", "alt_right"]
        and wide["canonical_residual"] == 5
    )
    check(ok, "高残差替代入带使必选/从不翻转为可选", f"HTTP {status} {wide}")

    # 场景 2：超大分层计数（60 个独立三点块，超额 e 计数 C(60,e)·2^(60-e)）。
    n = 180
    band_candidates = []
    for a in range(0, n, 3):
        band_candidates += [
            cand(f"z{a}", a, a + 2, 0),
            cand(f"a{a}", a, a + 1, 0),
            cand(f"b{a}", a + 1, a + 2, 1),
        ]
    payload = {"hits": hits(n), "candidates": band_candidates, "tolerance": 40}
    status, body = http_request("POST", "/audit-band", payload)
    from math import comb

    expected = {
        str(e): str(comb(60, e) * 2 ** (60 - e)) for e in range(41)
    }
    expected_total = str(sum(comb(60, e) * 2 ** (60 - e) for e in range(41)))
    ok = (
        status == 200
        and body["counts_by_excess"] == expected
        and body["total_count"] == expected_total
        and len(body["total_count"]) >= 28  # 29 位十进制，任意精度
        and "41" not in body["counts_by_excess"]
    )
    check(ok, "超大容差带分层计数精确（任意精度）", f"HTTP {status} counts mismatch")

    # 场景 3：零容差回归——与 /audit 关键字段完全一致。
    base = {
        "hits": hits(6),
        "candidates": [
            cand("bait", 0, 3, 0),
            cand("inner", 1, 2, 10),
            cand("tail", 4, 5, 1),
            cand("seq01", 0, 1, 1),
            cand("seq23", 2, 3, 1),
        ],
    }
    status, a = http_request("POST", "/audit", base)
    status_b, b = http_request("POST", "/audit-band", {**base, "tolerance": 0})
    ok = (
        status == 200
        and status_b == 200
        and b["counts_by_excess"] == {"0": a["optimal_count"]}
        and b["total_count"] == a["optimal_count"]
        and b["min_residual"] == a["total_residual"]
        and b["canonical_pairs"] == a["canonical_pairs"]
        and b["unmatched_hits"] == a["unmatched_hits"]
        and b["classification"] == a["classification"]
        and b["paired_hits"] == a["paired_hits"]
    )
    check(ok, "零容差结果与 /audit 完全一致", f"HTTP {status}/{status_b}")

    # 场景 4：非法 tolerance——字段路径 + 不夹带结果（含越界与类型错误）。
    for bad in (-1, 41, "5", 5.0, True):
        status, body = http_request(
            "POST", "/audit-band", {"hits": hits(4), "candidates": [], "tolerance": bad}
        )
        ok = (
            status == 400
            and set(body.keys()) == {"errors"}
            and any(e.get("field") == "/tolerance" for e in body["errors"])
        )
        check(ok, f"非法 tolerance 被拒绝: {bad!r}", f"HTTP {status} {body}")

    status, body = http_request(
        "POST", "/audit-band", {"hits": hits(4), "candidates": []}
    )
    check(
        status == 400 and any(e["field"] == "/tolerance" for e in body["errors"]),
        "缺少 tolerance 被拒绝", f"HTTP {status} {body}",
    )


def main() -> int:
    run_unit_tests()
    run_build_checks()
    run_smoke()
    run_band_smoke()

    print("\n=== 汇总 ===")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部复核通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
