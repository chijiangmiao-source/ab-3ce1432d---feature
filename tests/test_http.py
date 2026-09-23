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


# ----------------------------------------------------------- /audit-band


def band_payload(tolerance, candidates=None):
    return {
        "hits": [{"id": f"h{k}", "position": k} for k in range(4)],
        "candidates": candidates
        if candidates is not None
        else [
            {"id": "cheap_out", "left_endpoint": "h0", "right_endpoint": "h3", "residual": 0},
            {"id": "cheap_in", "left_endpoint": "h1", "right_endpoint": "h2", "residual": 0},
            {"id": "alt_left", "left_endpoint": "h0", "right_endpoint": "h1", "residual": 0},
            {"id": "alt_right", "left_endpoint": "h2", "right_endpoint": "h3", "residual": 5},
        ],
        "tolerance": tolerance,
    }


def test_audit_band_ok(server):
    status, body = request(server, "POST", "/audit-band", band_payload(5))
    assert status == 200
    assert body["tolerance"] == 5
    assert body["min_residual"] == 0
    assert body["band_residual_limit"] == 5
    assert body["counts_by_excess"] == {"0": "1", "5": "1"}
    assert body["total_count"] == "2"
    assert body["classification"]["required"] == []
    assert set(body["classification"]["optional"]) == {
        "cheap_out",
        "cheap_in",
        "alt_left",
        "alt_right",
    }
    assert [p["id"] for p in body["canonical_pairs"]] == ["alt_left", "alt_right"]
    assert body["canonical_residual"] == 5
    assert body["unmatched_hits"] == []


def test_audit_band_zero_matches_audit(server):
    for tolerance in (0, 4):
        status, body = request(server, "POST", "/audit-band", band_payload(tolerance))
        assert status == 200
        assert body["counts_by_excess"] == {"0": "1"}
        assert body["classification"]["required"] == ["cheap_in", "cheap_out"]

    status_a, audit_body = request(
        server,
        "POST",
        "/audit",
        {k: v for k, v in band_payload(0).items() if k != "tolerance"},
    )
    status_b, band_body = request(server, "POST", "/audit-band", band_payload(0))
    assert status_a == 200 and status_b == 200
    for key in ("optimal_count", "paired_hits", "total_residual", "canonical_pairs",
                "unmatched_hits", "classification"):
        if key == "total_residual":
            assert band_body["min_residual"] == audit_body[key]
        else:
            assert band_body[key] == audit_body[key]


@pytest.mark.parametrize("bad", [-1, 41, "5", 5.0, True, None])
def test_audit_band_invalid_tolerance(server, bad):
    status, body = request(server, "POST", "/audit-band", band_payload(bad))
    assert status == 400
    assert set(body.keys()) == {"errors"}
    assert any(e["field"] == "/tolerance" for e in body["errors"])
    for leaked in ("total_count", "min_residual", "canonical_pairs", "classification"):
        assert leaked not in body


def test_audit_band_missing_tolerance(server):
    payload = band_payload(0)
    del payload["tolerance"]
    status, body = request(server, "POST", "/audit-band", payload)
    assert status == 400
    assert any(e["field"] == "/tolerance" for e in body["errors"])


def test_audit_band_unknown_route_post(server):
    status, body = request(server, "POST", "/nope", {})
    assert status == 404
    assert body["errors"][0]["field"] == ""
