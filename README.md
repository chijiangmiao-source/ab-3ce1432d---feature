# 硅微条击中配对审计服务

把一列按位置严格递增的硅微条击中还原为互不交叉的粒子径迹配对，并在全部
最优方案上做审计：方案计数、规范解、未配对击中与每条候选对的必选/可选/从不分类。

- `POST /audit`：在唯一最低残差方案集上审计；
- `POST /audit-band`：在最大配对数下残差位于 `[R, R+tolerance]`（`tolerance`
  为 0–40 的整数）的全部方案构成的容差带上审计，避免把“唯一最低残差”过度
  解读为必选证据。

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

### `/audit-band` 容差带

请求沿用 `hits` / `candidates` 结构，新增整数 `tolerance`（0–40，越界或类型
错误返回字段路径 `/tolerance`，不夹带结果）。求解先确定最大配对数及其最低
残差 `R`，再纳入配对数相同且残差不超过 `R+tolerance` 的全部方案，响应：

- `min_residual`（即 R；`total_residual` 同义保留）、`band_residual_limit`：
  R 与 `R+tolerance`；
- `counts_by_excess`：按残差超额（相对 R）分组的方案数，键为超额整数、
  值为任意精度十进制字符串，仅列非零档；
- `total_count`（`optimal_count` 同义保留）：带内方案总数；
- `canonical_pairs` / `canonical_residual` / `unmatched_hits`：在整个容差带
  全部方案上，按左端位置顺序的配对 id 序列字典序裁决的规范解及其残差、未配对击中；
- `classification`：据整个容差带把每条候选判为 `required`（带内方案全部包含）、
  `optional`（部分包含）或 `never`（带内从不出现）。

`tolerance=0` 的结果与 `/audit` 完全一致（计数仅 `counts_by_excess["0"]`）。

## 算法

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数、最小残差、任意精度方案数，以及字典序最小 id 序列；
- outside DP（inside-outside）：统计每条候选弧出现在多少个最优方案中，
  据此分类 required / optional / never。

容差带审计不逐阈值重跑、也不枚举方案：标量 pass 先求出每个区间的最大对数与
最低残差 `R[i][j]`；谱 pass 维护相对 `R[i][j]` 的截断超额谱
`S[i][j][e]`（达到最大对数、残差恰为 `R[i][j]+e`、`0<=e<=tolerance` 的方案数），
以带移位的截断计数卷积精确合并——子方案超额非负且全局超额为各段之和，
按 tolerance 截断无损。outside 谱同样按父→子下发，候选弧在带内全部方案中的
全局出现量由“内子树谱 × 上下文谱（含外子树）前缀和”一次求得。
规范解在每预算 `0..tolerance` 上递推字典序最小 id 序列。

复杂度标量部分 O(n·|C| + n³)，谱部分与区间-弧 incidence 数成比例、每次
卷积至多 O(tolerance²)（n ≤ 180，|C| ≤ 4000，tolerance ≤ 40，最大规模实测 < 1 秒）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 求解（仅标准库）：`audit` / `audit_band` |
| `app.py` | HTTP 服务：`GET /health`、`POST /audit`、`POST /audit-band`（仅标准库） |
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

1. 代码测试（215 项，含 60 组 `/audit` 随机暴力对照与 80 组 `/audit-band`
   随机暴力对照、零容差回归、超大分层计数、非法 tolerance 等）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径、嵌套同优、交叉低价诱饵、空候选、非法引用、
   重复端点对、位置冲突、规模越界、未知路径）；
4. 容差带冒烟（高残差替代使归属变化、超大分层计数、零容差回归、非法 tolerance）。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
