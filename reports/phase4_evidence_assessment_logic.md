# Phase4 evidence assessment 评价逻辑说明

本文档说明 `Phase4-evidence_assessment.py` 的核心评价逻辑，重点解释 `comparator`、`outcome` 和 `paper` 三者的关系，以及 Phase4 在哪些环节调用 LLM。

## 1. Phase4 的输入从哪里来

Phase4 并不重新做题录筛选或全文纳入筛选，它读取 Phase3 全文评估后的两个核心结果：

- `paperinfo/*.json`：某个 comparator 下可用于证据评价的论文列表。
- `outcomeinfo/*.json`：某个 comparator 下、按 outcome 和 study design 组织好的待评价结局列表。

当前配置中 `pipeline.phase4_evidence_assessment.transfer_study_selection_files=true`，所以 Phase4 运行开始时会先把 Phase3 的 `paperinfo/outcomeinfo` 从：

```text
data/2021ACR RA/Study_Selection/
```

复制到：

```text
data/2021ACR RA/Evidence_Assessment/
```

随后 Phase4 实际读取的是 `Evidence_Assessment` 目录下的输入。

## 2. comparator 如何决定输入文件

`Phase4-evidence_assessment.py` 会从 Phase1 的 PICO 信息中读取 `C`，即 comparator 列表。当前 PICO `dff23ac6` 有两个 comparator：

```text
continue DMARDs at the same doses
abruptly withdraw DMARDs
```

脚本逐个 comparator 运行。每个 comparator 对应一组独立的 `paperinfo` 和 `outcomeinfo` 文件，文件后缀由 comparator 文本的 MD5 前 6 位生成：

```python
_c + md5(comparator)[:6]
```

当前实际文件关系是：

| comparator | postfix | paperinfo | outcomeinfo |
| --- | --- | --- | --- |
| `continue DMARDs at the same doses` | `_c590558` | `paperinfo_PICOdff23ac6_c590558.json` | `outcomeinfo_PICOdff23ac6_c590558.json` |
| `abruptly withdraw DMARDs` | `_c9b1431` | `paperinfo_PICOdff23ac6_c9b1431.json` | `outcomeinfo_PICOdff23ac6_c9b1431.json` |

也就是说，Phase4 的主循环是：

```text
for comparator in PICO["C"]:
    找到这个 comparator 对应的 postfix
    load_outcome_list(postfix)
    load_paper_list(postfix)
    assess_evidence(comparator)
```

## 3. comparator、outcome、paper 的关系

可以把它理解成三层结构：

```text
Comparator
  -> Outcome object
       -> related_paper_list
            -> Paper objects
```

更准确地说，Phase4 中的 `Outcome` 对象不是单纯的“一个 outcome 名称”，而是：

```text
comparator + outcome text + study design group
```

所以同一个 comparator 下，同一个 outcome 如果同时被 RCT 和 systematic review 支持，会生成两个不同的 `Outcome` 对象。它们的 `outcome` 文本一样，但 `assessment_results["GRADE"]["Study design"]` 不一样，`related_paper_list` 也不一样。

当前数据里最典型的例子是：

| comparator | outcome | study design | related papers |
| --- | --- | --- | --- |
| `continue DMARDs at the same doses` | `risk of disease flare` | `RANDOMIZED_CONTROLLED_TRIAL` | 7 |
| `continue DMARDs at the same doses` | `risk of disease flare` | `SYSTEMATIC_REVIEW` | 1 |

这就是为什么 `outcomeinfo_PICOdff23ac6_c590558.json` 里有 5 条 outcome，而不是 PICO 中配置的 4 个 outcome。因为 `risk of disease flare` 被拆成了两个 study design 分组。

当前实际数据统计：

| 文件 | 条目数 | 说明 |
| --- | ---: | --- |
| `paperinfo_PICOdff23ac6_c590558.json` | 14 | `continue DMARDs...` comparator 下纳入的 paper |
| `outcomeinfo_PICOdff23ac6_c590558.json` | 5 | 该 comparator 下按 outcome + study design 分组后的 outcome |
| `paperinfo_PICOdff23ac6_c9b1431.json` | 3 | `abruptly withdraw...` comparator 下纳入的 paper |
| `outcomeinfo_PICOdff23ac6_c9b1431.json` | 4 | 该 comparator 下按 outcome + study design 分组后的 outcome |

## 4. Phase3 如何生成这种关系

Phase3 全文评估完成后，会先为每个 comparator 找到匹配的 paper。匹配条件大致是：

```text
paper 的 population 匹配 PICO.P
paper 的 intervention 匹配 PICO.I
paper 的 comparator 匹配当前 comparator
```

匹配成功的 paper 会写入该 comparator 的 `paperinfo`。

然后 Phase3 再对当前 comparator 下的每个 outcome 做匹配：

```text
for outcome in PICO.O[comparator]:
    找到测量了该 outcome 的 paper
    按 paper.study_design 分组
    每个 study_design 分组生成一个 Outcome 对象
```

生成 `Outcome` 对象时，关键字段是：

- `comparator`：当前 comparator。
- `outcome`：当前 outcome 文本。
- `related_paper_list`：这个 comparator + outcome + study design 组合对应的 paper_uid 列表。
- `assessment_results["GRADE"]["Study design"]`：该 outcome 分组对应的研究设计。

因此，Phase4 不是“一个 paper 评价一次”，也不是“一个 outcome 评价一次”，而是：

```text
一个 comparator 下，一个 outcome 的一个 study design 证据组，评价一次。
```

## 5. Phase4 如何评价一个 comparator

Phase4 对每个 comparator 执行：

```python
quicker.load_outcome_list(comparator_postfix=input_postfix)
quicker.load_paper_list(comparator_postfix=input_postfix)
quicker.assess_evidence(comparator=comparator)
```

`quicker.assess_evidence()` 会创建一个 `Evidence` 对象，传入：

- 当前 comparator。
- 当前 comparator 对应的 outcome list。
- 当前 comparator 对应的 paper list。
- evidence assessment LLM。
- embeddings。
- GRADE 额外配置。

随后调用：

```python
evidence.assess_evidence()
```

它会逐个 `Outcome` 对象评价。

## 6. Phase4 如何把 outcome 和 paper 绑定起来

对每个 `Outcome`，`Evidence.assess_outcome()` 会用 `related_paper_list` 从当前 comparator 的 `paper_list` 里筛选相关 paper：

```text
related_paper = [
    p for p in paper_list
    if p.paper_uid in outcome.related_paper_list
]
```

然后按 `paper.study_design` 分组：

```text
study_design_group = {
    RANDOMIZED_CONTROLLED_TRIAL: [...papers],
    SYSTEMATIC_REVIEW: [...papers],
    ...
}
```

再根据 study design 调用不同评价函数：

| study design | 当前函数 | 状态 |
| --- | --- | --- |
| `RANDOMIZED_CONTROLLED_TRIAL` | `assess_rct()` | 已实现，走 GRADE |
| `SYSTEMATIC_REVIEW` | `assess_systematic_review()` | 未实现，跳过并记录状态 |
| `COHORT_STUDY` | `assess_cohort_study()` | 未实现，跳过并记录状态 |
| `META_ANALYSIS` | `assess_meta_analysis()` | 未实现，跳过并记录状态 |
| `OTHER_OBSERVATIONAL_STUDY` | `assess_cohort_study()` | 未实现，跳过并记录状态 |

你自己运行时报出的：

```text
Systematic review assessment has not been implemented yet
```

来源就是 `assess_systematic_review()`。这条信息的含义是：当前 outcome 的相关 paper 中存在 systematic review 分组，但项目还没有实现 systematic review 的证据评价方法。它不是 comparator/outcome 关系错误，而是研究设计评价分支尚未实现。

## 7. RCT 分支如何调用 LLM

当前真正完整实现的是 RCT 分支：

```text
Evidence.assess_rct()
  -> GRADEAssessment(...).run_assessment()
```

`GRADEAssessment.run_assessment()` 当前配置下主要做三件事：

1. 判断 outcome 的数据类型。
2. 如果配置 `extract_raw_data=true`，从每篇 paper 中抽取原始数据。
3. 根据 `factor_list` 做 GRADE 因素评价。

当前 `config/config.json` 中 Phase4 的 GRADE 配置是：

```json
{
  "factor_list": ["risk of bias"],
  "extract_raw_data": true
}
```

因此当前只评价 GRADE 的 `risk of bias`，同时会抽取 raw data。

## 8. LLM 调用点 1：判断 outcome 数据类型

函数：

```text
choose_data_type_of_outcome()
```

输入给 LLM 的核心信息是：

- population
- intervention
- comparator
- outcome
- disease

LLM 需要判断该 outcome 属于：

- `Dichotomous Data`
- `Continuous Data`
- `Time-to-Event Data`
- `Ordinal Data`
- `Count or Rate Data`
- `Not Applicable`

结果写入：

```text
outcome.assessment_results["GRADE"]["data type"]
```

## 9. LLM 调用点 2：从每篇 paper 抽取 raw data

如果 `extract_raw_data=true`，Phase4 会对该 outcome 的每篇相关 paper 调用：

```text
extract_data_for_paper()
```

对于 `Dichotomous Data`，会生成 6 个 cell 问题，例如：

- intervention 组总人数是多少？
- comparator 组总人数是多少？
- intervention 组发生目标 outcome 的人数是多少？
- intervention 组未发生目标 outcome 的人数是多少？
- comparator 组发生目标 outcome 的人数是多少？
- comparator 组未发生目标 outcome 的人数是多少？

每个 cell 的抽取大致分三步：

```text
原始 cell 问题
  -> LLM 生成更适合检索的 query 列表
  -> 用 query 到该 paper 的 Qdrant 向量库检索上下文
  -> LLM 基于上下文抽取数据和原文证据
  -> 对多个 query 的结果做去重/汇总
```

这里会大量调用 LLM 和 embedding 检索，所以 Phase4 会比 Phase3 后半段更慢。

结果写入：

```text
outcome.assessment_results["GRADE"]["Raw data from evidence"]
```

## 10. LLM 调用点 3：评价 RCT risk of bias

函数：

```text
assess_risk_of_bias_for_rcts()
```

它对每篇 RCT 先围绕以下问题做 RAG 分析：

- randomization process
- allocation concealment
- blinding
- incomplete accounting of patients and outcome events
- selective outcome reporting
- other limitations
- baseline risk of bias question

每篇 paper 的结果会被整理为一个中间 prompt。然后所有相关 RCT paper 的中间结果会被合并，交给 LLM 做一次总体 GRADE risk of bias 判断。

最终写入：

```text
outcome.assessment_results["GRADE"]["Risk of bias"]
```

结果结构大致是：

```json
{
  "result": "NOT_SERIOUS / SERIOUS / VERY_SERIOUS",
  "rationales": "..."
}
```

## 11. Qdrant 和 PDF 在 Phase4 中的角色

Phase4 不直接评价 PDF 文件本身，而是通过每篇 paper 的本地 Qdrant 向量库检索上下文。

关系是：

```text
Paper
  -> paper_uid
  -> data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/{paper_uid}_vector_database
  -> RAG 检索上下文
  -> LLM 评价或抽取
```

如果向量库已经存在且可读，Phase4 会加载本地 Qdrant 向量库。

如果向量库缺失、为空或不可读，Phase4 会尝试重建，这时才会再次用到 GROBID 和 PDF 解析。

## 12. 总结

Phase4 的评价单位不是单篇 paper，也不是单纯的 outcome 名称，而是：

```text
comparator + outcome + study_design_group
```

`paperinfo` 提供当前 comparator 下有哪些 paper。

`outcomeinfo` 提供当前 comparator 下有哪些 outcome 需要评价，以及每个 outcome 对应哪些 paper。

`related_paper_list` 是 `Outcome` 和 `Paper` 之间的关键连接字段。

LLM 主要用于：

- 判断 outcome 数据类型。
- 为 raw data 抽取生成检索 query。
- 基于 RAG 检索出的 paper 内容抽取数值和原文证据。
- 汇总 RCT 研究的 risk of bias。

当前代码真正完成的证据评价分支是 RCT 的 GRADE risk of bias 和 raw data 抽取。Systematic review、cohort、meta-analysis 等研究设计分支目前尚未实现，因此遇到这些分组时会跳过并记录状态。
