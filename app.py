"""硅微条击中配对审计 HTTP 服务（仅依赖 Python 标准库）。

路由：
* GET  /health   就绪探针；
* POST /audit    提交 {hits, candidates}，返回配对审计结果。

错误响应只包含 errors（每条带字段路径 field 与 message），不夹带任何审计字段。
监听端口由环境变量 PORT 控制（默认 8080）。
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from solver import ValidationError, audit

SERVICE_NAME = "track-pair-audit"
MAX_BODY_BYTES = 8 * 1024 * 1024


class AuditHandler(BaseHTTPRequestHandler):
    server_version = "TrackPairAudit/1.0"

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/health":
            self._write_json(
                HTTPStatus.OK,
                {"status": "ready", "service": SERVICE_NAME},
            )
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"errors": [{"field": "", "message": f"未知路径: {self.path}"}]},
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/audit":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"errors": [{"field": "", "message": f"未知路径: {self.path}"}]},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"errors": [{"field": "", "message": "Content-Length 非法"}]},
            )
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"errors": [{"field": "", "message": "请求体为空或超过大小限制"}]},
            )
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"errors": [{"field": "", "message": f"请求体不是合法 JSON: {exc}"}]},
            )
            return

        try:
            result = audit(payload)
        except ValidationError as exc:
            # 错误响应不夹带任何审计结果。
            self._write_json(HTTPStatus.BAD_REQUEST, {"errors": exc.errors})
            return
        self._write_json(HTTPStatus.OK, result)

    def log_message(self, fmt: str, *args: object) -> None:
        # 保持容器日志简洁；正常由上层采集。
        print(f"[http] {self.address_string()} {fmt % args}")


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AuditHandler)
    print(f"[http] {SERVICE_NAME} listening on 0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
