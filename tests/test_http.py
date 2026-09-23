"""HTTP 层测试：在随机端口上线程内启动真实 app，走 urllib 调用。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from app import AuditHandler


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), AuditHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def request(base, method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health(server):
    status, body = request(server, "GET", "/health")
    assert status == 200
    assert body == {"status": "ready", "service": "track-pair-audit"}


def test_audit_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 2},
            {"id": "q", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 1},
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 200
    assert body["optimal_count"] == "1"
    assert [p["id"] for p in body["canonical_pairs"]] == ["p", "q"]
    assert body["unmatched_hits"] == []


def test_audit_error_shape_has_no_audit_fields(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "x", "left_endpoint": "h0", "right_endpoint": "nope", "residual": 0}
        ],
    }
    status, body = request(server, "POST", "/audit", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"


def test_bad_json(server):
    req = urllib.request.Request(
        server + "/audit", data=b"{not json", method="POST"
    )
    req.add_header("Content-Type", "application/json")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_unknown_route(server):
    status, _ = request(server, "GET", "/")
    assert status == 404


# ---------------------------------------------------------------- /audit-band


def test_audit_band_ok(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "low", "left_endpoint": "h0", "right_endpoint": "h1", "residual": 0},
            {"id": "rest", "left_endpoint": "h2", "right_endpoint": "h3", "residual": 0},
            {"id": "z_alt", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 5},
            {"id": "alt_rest", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 0},
        ],
        "tolerance": 5,
    }
    status, body = request(server, "POST", "/audit-band", payload)
    assert status == 200
    assert body["minimum_residual"] == 0
    assert body["band_residual_limit"] == 5
    assert body["band_count"] == "2"
    counts = {row["excess"]: row["count"] for row in body["residual_bands"]}
    assert counts[0] == "1"
    assert counts[5] == "1"
    assert set(body["classification"]["optional"]) == {
        "low", "rest", "z_alt", "alt_rest"
    }
    assert [p["id"] for p in body["canonical_pairs"]] == ["low", "rest"]
    assert body["canonical_residual"] == 0
    assert body["unmatched_hits"] == []


def test_audit_band_zero_matches_audit(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "p", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 2},
            {"id": "q", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 1},
        ],
    }
    s1, audit_body = request(server, "POST", "/audit", payload)
    s2, band_body = request(server, "POST", "/audit-band", {**payload, "tolerance": 0})
    assert (s1, s2) == (200, 200)
    assert band_body["band_count"] == audit_body["optimal_count"]
    assert band_body["minimum_residual"] == audit_body["total_residual"]
    assert band_body["canonical_pairs"] == audit_body["canonical_pairs"]
    assert band_body["unmatched_hits"] == audit_body["unmatched_hits"]
    assert band_body["classification"] == audit_body["classification"]


@pytest.mark.parametrize("bad_tolerance", [41, -1, "5", 1.0, True, None])
def test_audit_band_bad_tolerance(server, bad_tolerance):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [],
        "tolerance": bad_tolerance,
    }
    status, body = request(server, "POST", "/audit-band", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert any(e["field"] == "/tolerance" for e in body["errors"])
    assert "band_count" not in body and "minimum_residual" not in body


def test_audit_band_missing_tolerance(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [],
    }
    status, body = request(server, "POST", "/audit-band", payload)
    assert status == 400
    assert any(e["field"] == "/tolerance" for e in body["errors"])
    assert set(body.keys()) == {"errors"}


def test_audit_band_validation_error_shape(server):
    payload = {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": [
            {"id": "x", "left_endpoint": "h0", "right_endpoint": "nope", "residual": 0}
        ],
        "tolerance": 3,
    }
    status, body = request(server, "POST", "/audit-band", payload)
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert body["errors"][0]["field"] == "/candidates/0/right_endpoint"
