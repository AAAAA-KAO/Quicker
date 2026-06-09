# hybrid_retrieval.py 与 llm_judge_router.py 结构化输出分析报告

> 生成日期: 2026-06-09
> 分析目标: 明确两个核心脚本的输入/输出接口,便于下游流水线集成

---

## 1. `src/hybrid_retrieval.py` — 混合检索结构化输出

### 1.1 核心函数签名

| 函数 | 签名 | 说明 |
|------|------|------|
| `hybrid_retrieve()` | `(question: str, pico: dict, config?: RetrievalConfig) -> dict[str, list[SearchResult]]` | 顶层入口,一次性执行稠密+稀疏+混合三路检索 |
| `ClinicalQAHybridRetriever.retrieve()` | `(self, question: str, pico: dict) -> dict[str, list[SearchResult]]` | 可复用检索器实例方法,内部逻辑同上 |

### 1.2 返回值的顶层结构

```python
{
    "dense":   list[SearchResult],   # Qdrant 稠密向量检索的前 Top-K 结果
    "sparse":  list[SearchResult],   # BM25 稀疏检索的前 Top-K 结果
    "hybrid":  list[SearchResult],   # 加权融合后的前 Top-K 结果
}
```

- 默认 `top_k = 5`,即每个列表至多包含 5 条结果。
- 融合权重默认为 `dense_weight = 0.6`、`sparse_weight = 0.4`,对两路分数做最大最小归一化后线性加权。

### 1.3 `SearchResult` 数据类 (lines 79–85)

```python
@dataclass(frozen=True)
class SearchResult:
    index:  int              # 原始知识库 records 列表中的索引,用于跨路对齐
    score:  float            # 该条结果的检索分数 (dense 为向量相似度, sparse 为 BM25 得分, hybrid 为加权融合分)
    record: dict[str, Any]   # 完整的知识库条目字典
```

### 1.4 `record` 字典的典型字段

`record` 来源于 BM25 pickle 文件中的 `records` 列表,并结合 Qdrant payload 合并。典型字段如下:

| 字段 | 类型 | 说明 |
|------|------|------|
| `question_id` | `str` / `int` | 知识库中该问答对的唯一标识 |
| `question` | `str` | 临床问题原文 (如 "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?") |
| `answer` | `str` 或 `list[str]` | 临床指南答案; 若为列表则用 `" \| "` 连接打印 |
| `disease` | `str` | 相关疾病或临床领域 (如 "Appendicitis") |
| `topic` | `str` | 临床主题分类 |
| `pico` | `dict` | 该条目的 PICO 结构 (P/I/C/O) |
| `source` | `dict` | 来源信息 (指南名称、章节等) |
| `synonyms` | `list` 或其他 | 同义词/相关词列表 |
| `search_text` | `str` | 建库时的检索文本 |

### 1.5 终端打印格式 (via `print_results()`)

```
=== Qdrant Dense Top Results ===

[1] score=0.8521
question: Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?
answer: In pediatric patients with suspected appendicitis, clinical scores alone are insufficient...
disease: Appendicitis

[2] score=0.7834
...
```

### 1.6 输出不落盘

该脚本**不**将检索结果写入文件。返回值仅通过以下方式暴露:
- 函数返回值 (供 `import` 调用);
- 终端 `print` 输出 (供命令行使用)。

---

## 2. `src/llm_judge_router.py` — LLM 智能路由结构化输出

### 2.1 核心函数签名

| 函数 | 签名 | 说明 |
|------|------|------|
| `judge_route()` | `(question, pico, retrieved_qa_pairs, llm, ...) -> dict[str, Any]` | 单条临床问题的路由判断入口 |
| `judge_route_batch()` | `(llm, cases, max_concurrency) -> list[dict[str, Any]]` | 批量路由判断,通过 LangChain `llm.batch()` 并发调用 |

### 2.2 返回值的顶层结构

```python
{
    "判断":              str,              # "yes" 或 "no"
    "理由":              str,              # 最终判断的简要理由
    "维度理由": {
        "检索匹配强度":   str,              # 该维度的详细理由
        "候选答案一致性": str,              # 该维度的详细理由
        "PICO覆盖度":    str,              # 该维度的详细理由
    },
    "维度评级": {
        "检索匹配强度":   str,              # 枚举值: "强" / "中" / "弱"
        "候选答案一致性": str,              # 枚举值: "一致" / "部分一致" / "冲突" / "不足"
        "PICO覆盖度":    str,              # 枚举值: "完整" / "部分" / "不足"
    },
    "依据候选排名":       list[int],         # 支撑判断结论的候选 QA 排名列表,如 [1, 3]
    "基于候选的简短答案":  str,              # 当 判断="yes" 时基于候选的简要回答; 否则为空字符串
}
```

### 2.3 各字段详解

#### 2.3.1 `判断` (str: "yes" | "no")

LLM judge 的最终路由决策。**保守策略**: 只有当检索到的 QA 对明确匹配临床问题、候选答案连贯且覆盖关键 PICO 要素时才返回 `"yes"`。

#### 2.3.2 `理由` (str)

用一句话概括最终判断的核心依据,便于人工审阅。

#### 2.3.3 `维度理由` (dict, 3 个 key)

三个评估维度各自的自然语言理由:

| 维度 | 评估内容 |
|------|----------|
| **检索匹配强度** | 候选问答的问题、疾病、主题、检索分数是否与临床问题匹配 |
| **候选答案一致性** | 候选答案之间是否支持同一结论,是否有冲突或仅间接相关 |
| **PICO覆盖度** | 候选问答是否覆盖输入 PICO 中的人群(P)、干预(I)、对照(C)、结局(O) |

#### 2.3.4 `维度评级` (dict, 3 个 key)

每个维度的结构化等级判定,枚举值如下:

| 维度 | 可选值 |
|------|--------|
| 检索匹配强度 | `"强"` / `"中"` / `"弱"` |
| 候选答案一致性 | `"一致"` / `"部分一致"` / `"冲突"` / `"不足"` |
| PICO覆盖度 | `"完整"` / `"部分"` / `"不足"` |

#### 2.3.5 `依据候选排名` (list[int])

支撑 `判断` 结论的具体候选条目排名编号。例如 `[1, 2]` 表示排名第 1 和第 2 的候选用作判断依据。此排名编号对应输入候选列表的 1-based rank。

#### 2.3.6 `基于候选的简短答案` (str)

- 若 `判断 = "yes"`: 基于候选 QA 对提炼出的简短临床回答。
- 若 `判断 = "no"`: 为空字符串 `""`。

### 2.4 输出格式

终端打印**纯 JSON** (无 markdown 代码块包裹),通过 `json.dumps(result, ensure_ascii=False, indent=2)` 输出。不落盘。

### 2.5 输入兼容性

该脚本可接收三种检索结果格式 (通过 `normalize_retrieval_results()` 统一规范化):

| 格式 | 示例 |
|------|------|
| 混合检索结果字典 | `{"dense": [...], "sparse": [...], "hybrid": [...]}` |
| SearchResult 字典列表 | `[{"index": 0, "score": 0.82, "record": {...}}]` |
| 知识库 record 字典列表 | `[{"question_id": ..., "question": ..., "answer": ...}]` |

默认使用 `--retrieval-method hybrid` 从第一种格式中选取 hybrid 列表。

---

## 3. 两脚本的数据流对接关系

```
用户输入 (临床问题 + PICO)
    │
    ▼
┌─────────────────────────────────┐
│  hybrid_retrieval.py            │
│  hybrid_retrieve()              │
│                                 │
│  输出:                          │
│  {                              │
│    "dense":  [SearchResult×5],  │
│    "sparse": [SearchResult×5],  │
│    "hybrid": [SearchResult×5]   │  ──── 可直接作为 ────►
│  }                              │
└─────────────────────────────────┘
                                          │
                                          │ --retrieval-file / --retrieval-json
                                          ▼
                                 ┌─────────────────────────────────┐
                                 │  llm_judge_router.py            │
                                 │  judge_route()                  │
                                 │                                │
                                 │  输出:                          │
                                 │  {                              │
                                 │    "判断": "yes" | "no",        │
                                 │    "理由": "...",               │
                                 │    "维度理由": {...},            │
                                 │    "维度评级": {...},            │
                                 │    "依据候选排名": [...],        │
                                 │    "基于候选的简短答案": "..."    │
                                 │  }                              │
                                 └─────────────────────────────────┘
                                          │
                                          ▼
                                    下游决策 / 生成模块
```

### 实际对接示例

```bash
# Step 1: 混合检索并保存结果
conda run -n quicker python src/hybrid_retrieval.py \
    --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?" \
    --pico-json '{"P":"pediatric patients with suspected appendicitis","I":"clinical scores alone","C":["imaging or laboratory-assisted diagnosis"],"O":{"imaging or laboratory-assisted diagnosis":["diagnostic accuracy","missed appendicitis"]}}' \
    > results/retrieval_output.json

# Step 2: 将检索结果送入 LLM judge 进行智能路由
conda run -n quicker python src/llm_judge_router.py \
    --question "Should pediatric patients with suspected appendicitis be diagnosed by clinical scores alone?" \
    --pico-json '{"P":"pediatric patients with suspected appendicitis","I":"clinical scores alone","C":["imaging or laboratory-assisted diagnosis"],"O":{"imaging or laboratory-assisted diagnosis":["diagnostic accuracy","missed appendicitis"]}}' \
    --retrieval-file results/retrieval_output.json
```

### Python API 对接示例

```python
from hybrid_retrieval import hybrid_retrieve, RetrievalConfig
from llm_judge_router import judge_route, build_llm

# Step 1: 检索
config = RetrievalConfig(top_k=5)
retrieval_results = hybrid_retrieve(question, pico, config)

# Step 2: 路由 — 直接将 hybrid_retrieve 的返回值传入 judge_route
llm = build_llm(model="deepseek-v4-flash", api_key="...", base_url="...", temperature=0)
route_result = judge_route(
    question=question,
    pico=pico,
    retrieved_qa_pairs=retrieval_results,  # 直接对接
    llm=llm,
    retrieval_method="hybrid",
)
# route_result 即为上述结构化路由字典
```

---

## 4. 总结

| 维度 | `hybrid_retrieval.py` | `llm_judge_router.py` |
|------|----------------------|------------------------|
| **功能定位** | 多路检索 + 分数融合 | LLM 三维修判断 + 智能路由 |
| **输出类型** | `dict[str, list[SearchResult]]` | `dict[str, Any]` (规范化 JSON) |
| **输出 key 数量** | 3 ("dense", "sparse", "hybrid") | 6 (判断/理由/维度理由/维度评级/依据候选排名/基于候选的简短答案) |
| **核心产出** | 排序后的候选 QA 对列表 | "yes/no" 路由决策 + 三维度评估 |
| **是否落盘** | 否 (仅终端打印 + 函数返回) | 否 (仅终端打印 + 函数返回) |
| **下游用途** | 为路由/生成提供候选证据 | 决定是否直接回答 (yes) 或进入更复杂的生成流程 (no) |
| **是否可编程调用** | 是 (`hybrid_retrieve()`) | 是 (`judge_route()` / `judge_route_batch()`) |
