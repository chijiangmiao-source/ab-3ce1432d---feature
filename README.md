# 硅微条击中配对审计服务

把一列按位置严格递增的硅微条击中还原为互不交叉的粒子径迹配对，并在全部
最优方案上做审计：方案计数、规范解、未配对击中与每条候选对的必选/可选/从不分类。

## 问题与目标

- 输入：4–180 个击中（`id` 唯一、`position` 严格递增），至多 4000 条候选配对
  （`id` 唯一、端点对不重复、`residual` 为非负整数）。
- 选中的配对必须端点互异，按位置绘制后两两不交叉（允许嵌套）。
- 优化顺序（字典序）：
  1. 最大化已配对击中数（即最大化配对对数）；
  2. 最小化残差总和；
  3. 以“左端位置顺序下的配对 id 序列”字典序最小者为规范解。
- 审计输出：
  - `optimal_count`：任意精度十进制（字符串承载），达到前两级目标的方案数；
  - `canonical_pairs`：规范配对（含端点与残差，按左端位置顺序）；
  - `unmatched_hits`：规范解中的未配对击中；
  - `classification`：依据全部最优方案给出 `required` / `optional` / `never`。
- 空候选列表合法，返回计数 1 的唯一空方案。
- 重复端点对、未知端点、位置不严格递增/冲突、标识重复、规模越界等均返回
  `400 {"errors": [{"field": "/路径", "message": "..."}]}`，错误响应不夹带任何审计字段。

## 算法

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数、最小残差、任意精度方案数，以及字典序最小 id 序列；
- outside DP（inside-outside）：统计每条候选弧出现在多少个最优方案中，
  据此分类 required / optional / never。

复杂度 O(n·|C| + n³)（n ≤ 180，|C| ≤ 4000），最大规模实测约 0.2 秒。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 求解（仅标准库） |
| `app.py` | HTTP 服务：`GET /health`、`POST /audit`（仅标准库） |
| `verify.py` | 单次复核：pytest、构建检查、API/HTTP 冒烟 |
| `tests/` | 单元测试、进程内 HTTP 测试、随机暴力枚举交叉验证 |
| `Dockerfile` | API 镜像定义 |
| `docker-compose.yml` | `api` 服务 + `verify` 复核服务 |

## 运行

```bash
# 默认宿主机端口 8080，可用 HOST_PORT 覆盖
HOST_PORT=9090 docker compose up --build -d api

curl -s http://localhost:9090/health
# {"status":"ready","service":"track-pair-audit"}
```

审计请求示例：

```bash
curl -s -X POST http://localhost:9090/audit \
  -H 'Content-Type: application/json' \
  -d '{
    "hits": [
      {"id":"h0","position":0},{"id":"h1","position":10},
      {"id":"h2","position":20},{"id":"h3","position":30}
    ],
    "candidates": [
      {"id":"a_out","left_endpoint":"h0","right_endpoint":"h3","residual":1},
      {"id":"a_in","left_endpoint":"h1","right_endpoint":"h2","residual":5},
      {"id":"b_left","left_endpoint":"h0","right_endpoint":"h1","residual":3},
      {"id":"b_right","left_endpoint":"h2","right_endpoint":"h3","residual":3}
    ]
  }'
```

## 复核（verify 单次服务）

```bash
docker compose build && docker compose run --rm verify
```

`verify` 服务等待 `api` 健康后依次执行：

1. 代码测试（77 项，含 60 组随机输入与暴力枚举的方案数/规范解/分类对照）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径、嵌套同优、交叉低价诱饵、空候选、非法引用、
   重复端点对、位置冲突、规模越界、未知路径）。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
