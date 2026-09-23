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

## 容差带审计：`POST /audit-band`

`/audit` 只承认唯一最低残差方案，容易把"次优但仍可接受"的替代配错误判为
`never`。`/audit-band` 在同一区间分解上一次性回答整个容差带：

请求在原结构上加入整数 `tolerance`（`0–40`，必填；越界或类型错误返回
`400` 且 `field` 为 `/tolerance`，不夹带任何结果）：

```json
{"hits": [...], "candidates": [...], "tolerance": 5}
```

语义：先求最大配对数及其最低残差 `R`，再纳入配对数相同、残差不超过
`R+tolerance` 的**全部**方案。响应：

- `minimum_residual`：R；`band_residual_limit`：R+tolerance；
- `residual_bands`：按残差超额 d=0..tolerance 逐档分组的方案数
  `[{excess, residual: R+d, count}]`，`count` 为任意精度十进制字符串，
  不可达的超额档位显式给出 `"0"`；
- `band_count`：带内总方案数（任意精度十进制字符串）；
- `canonical_pairs` / `canonical_residual` / `unmatched_hits`：
  以"左端位置顺序下的配对 id 序列"字典序裁决的带内规范解（可能来自非零
  超额层——当其首标识更小）及其残差、未配对击中；
- `classification`：据**整个容差带**把每条候选判为 `required`（带内每个
  方案都出现）、`optional`（部分出现）或 `never`（带内从不出现）。

`tolerance=0` 的结果与 `/audit` 完全一致（同样的计数、规范解、未配对击中
与三分类）。算法不按残差阈值重跑、也不逐个枚举方案：区间 DP 对每个区间
维护"相对局部最优残差的截断超额谱"（只保留超额 0..tolerance），
inside 合并做截断卷积，outside 传播兄弟谱，候选全局出现量由
内外谱对所有父区间一次性求和得到；复杂度
O(n·|C| + n³ + n²·t + n·|C|·t²)，n ≤ 180、|C| ≤ 4000、t ≤ 40，
最大规模实测约 0.5 秒。

## 算法（区间分解）

区间非交叉匹配（允许嵌套），状态为区间 `[i,j)`：

- inside DP：`跳过 i` 或 `i 与 k 配对`（内部 `[i+1,k)` 与外部 `[k+1,j)` 独立），
  维护最大对数、最小残差、任意精度方案数，以及字典序最小 id 序列；
- 容差带在此之上为每个区间维护截断残差谱 `S[i][j][d]`：配对数最优的规则
  把全局超额可加地分解为 `δ = 规则残差 + Σ子区间最优残差 - 父区间最优残差`，
  规则树 δ 望远镜求和恰为总残差超额，故谱合并是精确的截断卷积；
- outside DP（inside-outside）：零容差传播方案数，容差带传播截断谱，
  统计每条候选弧出现在多少个（带内）最优方案中，据此分类
  required / optional / never。

复杂度 O(n·|C| + n³)（n ≤ 180，|C| ≤ 4000），最大规模实测约 0.2 秒；
容差带额外开销随 tolerance 线性/平方增长（见上）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `solver.py` | 校验 + 区间 DP 求解（含零容差与截断残差谱容差带，仅标准库） |
| `app.py` | HTTP 服务：`GET /health`、`POST /audit`、`POST /audit-band`（仅标准库） |
| `verify.py` | 单次复核：pytest、构建检查、API/HTTP 冒烟 |
| `tests/` | 单元测试、进程内 HTTP 测试、随机暴力枚举交叉验证（含容差带） |
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

容差带请求示例（`tolerance=5`，纳入残差不超过 R+5 的全部最大配对方案）：

```bash
curl -s -X POST http://localhost:9090/audit-band \
  -H 'Content-Type: application/json' \
  -d '{"hits":[...],"candidates":[...],"tolerance":5}'
```

## 复核（verify 单次服务）

```bash
docker compose build && docker compose run --rm verify
```

`verify` 服务等待 `api` 健康后依次执行：

1. 代码测试（187 项，含随机输入与暴力枚举的方案数/规范解/分类/候选出现量
   对照，覆盖零容差与容差带）；
2. 构建检查（语法编译、模块导入、镜像内关键文件齐备）；
3. API/HTTP 冒烟（健康路径、嵌套同优、交叉低价诱饵、空候选、非法引用、
   重复端点对、位置冲突、规模越界、未知路径，以及容差带的高残差替代
   归属变化、超大分层计数、零容差回归、非法/缺失容差）。

全部通过退出码 0，任一失败非零。

## 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
PORT=8080 python app.py
```
