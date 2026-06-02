# Quicker Phase1-Phase5 集成运行实验报告

## 1. 实验概览

- 数据集：`2021ACR RA`
- 疾病/主题：`Rheumatoid Arthritis (RA)`
- 临床问题：`Should patients with RA on DMARDs who are in low disease activity gradually taper off DMARDs, abruptly withdraw DMARDs, or continue DMARDS at the same doses?`
- PICO 索引：`dff23ac6`
- 集成入口脚本：`main.py`
- 运行配置文件：`config/config.json`
- 运行清单：`reports/run_manifest_dff23ac6.json`
- 最新推荐结果输出：`data/2021ACR RA/Recommendation_Formation/quicker_data(PICO_IDXdff23ac6)_20260601204748.json`

本次集成将 Phase1 到 Phase5 串联到同一个 Python 入口中运行。单独的 `Phase3-study_selection(full-text_assessment only).py` 未纳入集成流程，因为常规 Phase3 工作流已经包含全文评估功能。

## 2. 模型与运行配置

所有运行参数均集中放置在 `config/config.json` 中。

- 所有阶段使用的 LLM 模型：`qwen-plus`
- API base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- API key：已配置在 `config/config.json` 中，本报告中不展示明文
- Embedding 模型：`text-embedding-v4`
- 文献初筛方法：`basic`
- 全文评估方法：`RAG`
- 文献初筛重复次数：`3`
- 进入全文评估的纳入阈值：`2`
- Phase4 meta-analysis：默认关闭；该部分属于可选统计演示流程，并不是本 RA 临床问题链路的必要下游步骤

## 3. 集成工作流

### Phase1：问题分解

目标：将临床问题转换为结构化 PICO 信息。

步骤：

1. 从 `config/config.json` 读取临床问题、数据集名称、模型设置和输出路径。
2. 当 `pico_idx` 为 `auto` 时，根据 `clinical_question + dataset_name` 计算稳定的 PICO 索引。
3. 使用问题分解模型抽取：
   - Population
   - Intervention
   - Comparators
   - 按 comparator 映射的 Outcomes
4. 保存或更新 PICO 记录。

输入：

- `config/config.json`
- `pipeline.clinical_question` 中配置的临床问题

输出：

- `data/2021ACR RA/Question_Decomposition/PICO_Information.json`

本次运行状态：

- 已跳过，因为复用了已有 PICO `dff23ac6`。

### Phase2：文献检索

目标：生成 PubMed 检索策略，检索 PubMed 文献记录，并过滤不可用记录。

步骤：

1. 从 Phase1 输出中读取 PICO `dff23ac6`。
2. 使用文献检索模型生成检索词和 PubMed 检索策略。
3. 应用配置中的 PubMed 参数和过滤器：
   - 发表日期范围：`1946` 到 `2025/03/30`
   - RCT 聚焦过滤器
   - 排除 review 的过滤器
4. 检索 PubMed 记录。
5. 移除没有摘要的记录。
6. 按 `Paper_Index` 去重。
7. 移除配置中指定的无效文献类型。
8. 保存供后续阶段使用的 Quicker 数据包。

输入：

- `data/2021ACR RA/Question_Decomposition/PICO_Information.json`
- `config/config.json` 中的检索设置

输出：

- 原始检索策略：`data/2021ACR RA/Literature_Search/pubmed/Results/{model_name}/use_agent_{True|False}/PICO{pico_idx}_search_strategy.txt`
- 原始 PubMed 检索结果：`data/2021ACR RA/Literature_Search/pubmed/Results/{model_name}/use_agent_{True|False}/PICO{pico_idx}.json`
- 过滤后的下游数据包：`data/2021ACR RA/quicker_data(PICO_IDXdff23ac6)_ls.json`

本次运行状态：

- 已跳过，因为过滤后的下游数据包已存在。

### Phase3：研究筛选

目标：对检索记录进行题录/摘要筛选，并对潜在纳入研究执行全文评估。

步骤：

1. 加载 Phase2 生成的过滤后检索数据包。
2. 构建包含 PICO 信息的临床问题上下文，用于文献筛选。
3. 使用配置的方法和重复次数执行题录/摘要筛选。
4. 汇总多次筛选投票结果。
5. 选择纳入票数达到阈值的记录进入全文评估。
6. 在全文评估前检查本地是否已存在所需 PDF。
7. 如果缺少 PDF，则停止运行，并在 `reports/` 下写入 PDF 请求文件。
8. 如果 PDF 已存在，则基于 RAG 从全文中抽取 PICO/研究特征。
9. 保存纳入研究元数据和 outcome 层面的研究筛选输出。

输入：

- `data/2021ACR RA/quicker_data(PICO_IDXdff23ac6)_ls.json`
- `config/config.json`
- 本地 PDF：`data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/`

输出：

- 筛选 CSV：`data/2021ACR RA/Study_Selection/Results/screening_records/basic/dff23ac6/`
- 研究筛选阶段的 paper metadata：`data/2021ACR RA/Study_Selection/paperinfo/paperinfo_PICOdff23ac6*.json`
- 研究筛选阶段的 outcome metadata：`data/2021ACR RA/Study_Selection/outcomeinfo/outcomeinfo_PICOdff23ac6*.json`

本次运行状态：

- 已跳过，因为复用了已有研究筛选输出。

### Phase4：证据评价

目标：针对已筛选出的 outcomes 评价证据确定性和 GRADE 相关信息。

步骤：

1. 可选地将 Phase3 的 `outcomeinfo` 和 `paperinfo` 文件转移到证据评价目录。
2. 针对每个配置的 comparator，加载 PICO、outcome 和 paper metadata。
3. 在证据评价前检查本地是否已存在所需 PDF。
4. 如果缺少 PDF，则停止运行，并在 `reports/` 下写入 PDF 请求文件。
5. 如果 PDF 已存在，则创建或复用用于论文检索的向量库。
6. 抽取 outcome 相关的原始证据信息。
7. 评价配置中指定的 GRADE 因素。
8. 保存已评价的 outcome 和 paper 信息。

输入：

- `data/2021ACR RA/Study_Selection/outcomeinfo/outcomeinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Study_Selection/paperinfo/paperinfo_PICOdff23ac6_c844628.json`
- PDF：`data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/`
- `config/config.json` 中的 GRADE 要求

输出：

- `data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6.json`
- `data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Evidence_Assessment/paperinfo/paperinfo_PICOdff23ac6_c844628.json`

本次运行状态：

- 已跳过，因为已存在经过证据评价的输出。

### Phase4 可选项：Meta-Analysis

目标：在配置启用时运行统计学 meta-analysis 示例。

步骤：

1. 读取配置中的二分类或连续型 outcome 数据集。
2. 在可用时使用 `rpy2` 和 R 的 `meta` 包进行分析。
3. 将文本摘要保存到 `reports/meta_analysis/{pico_idx}/`。

输入：

- `pipeline.phase4_meta_analysis.binary_outcomes`
- `pipeline.phase4_meta_analysis.continuous_outcomes`

输出：

- `reports/meta_analysis/{pico_idx}/binary_meta_*.txt`
- `reports/meta_analysis/{pico_idx}/continuous_meta_*.txt`

本次运行状态：

- 已在配置中关闭。

### Phase5：推荐形成

目标：将证据评价结果综合为最终临床推荐文本。

步骤：

1. 将证据评价阶段的 `outcomeinfo` 和 `paperinfo` 文件转移到推荐形成目录。
2. 加载 PICO `dff23ac6` 的已评价 outcome 信息。
3. 解释每个 outcome 的 evidence profile。
4. 按 comparator 汇总 outcomes。
5. 综合推荐依据。
6. 生成最终推荐。
7. 保存完整推荐结果包。

输入：

- `data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6.json`
- `data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Evidence_Assessment/paperinfo/paperinfo_PICOdff23ac6_c844628.json`
- 配置中的总体证据确定性：`LOW`

输出：

- `data/2021ACR RA/Recommendation_Formation/quicker_data(PICO_IDXdff23ac6)_20260601204748.json`

本次运行状态：

- 已使用 `qwen-plus` 完成。

推荐摘要：

生成的推荐建议：对于已经达到并维持低疾病活动度的成人 RA 患者，应继续以当前剂量使用 DMARDs；不建议常规减量或突然停药。若在充分共同决策后尝试减量，则应采用谨慎且密切监测的方案。

## 4. PDF 处理机制

集成后的流程会在进入依赖全文的步骤前检查本地 PDF 是否存在。

期望 PDF 路径格式：

- `data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/{paper_uid}.pdf`

如果缺少必需 PDF，流程会在尝试下载或全文分析前停止，并写入：

- `reports/missing_pdfs_{stage}_PICO{pico_idx}.json`
- `reports/missing_pdfs_{stage}_PICO{pico_idx}.md`

每个缺失 PDF 报告包含：

- Paper UID
- 标题
- PMID
- DOI
- 精确文件夹路径
- 推荐 PDF 文件路径

手动将 PDF 放入指定目录后，重新运行：

```bash
python main.py
```

## 5. 代码与配置修改

修改或新增的文件：

- `main.py`
- `config/config.json`
- `config/config-template.json`
- `reports/run_manifest_dff23ac6.json`
- `reports/experiment_report_dff23ac6.md`

主要集成修改：

- 新增统一项目入口 `main.py`。
- 将所有阶段路径、模型设置、阶段开关、检索参数、研究筛选参数、证据评价参数、推荐形成参数和 PDF 处理设置集中到 `config/config.json`。
- 复用已有 Phase1-Phase5 工具模块，避免重复实现核心逻辑。
- 未纳入 `Phase3-study_selection(full-text_assessment only).py`，因为常规 Phase3 已包含全文评估。
- 在 `reports/` 下新增运行清单生成。
- 在全文评估和证据评价前新增缺失 PDF 预检查机制。
- 接入可选 Phase4 meta-analysis，默认关闭。

配置修改：

- 所有阶段 LLM 均配置为 `qwen-plus`。
- 配置 OpenAI-compatible DashScope base URL。
- 配置 embedding 模型为 `text-embedding-v4`。
- 新增 `pipeline.paths` 管理所有输入/输出目录。
- 新增 `pipeline.stages` 控制每个阶段是否启用。
- 新增 `pipeline.pdf_handling`，用于在缺少 PDF 时停止并报告。
- 新增 `pipeline.phase1_*` 到 `pipeline.phase5_*` 的阶段级运行参数。

## 6. 当前输出文件清单

本次运行产生或复用的关键文件：

- `data/2021ACR RA/Question_Decomposition/PICO_Information.json`
- `data/2021ACR RA/quicker_data(PICO_IDXdff23ac6)_ls.json`
- `data/2021ACR RA/Study_Selection/outcomeinfo/outcomeinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Study_Selection/paperinfo/paperinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6.json`
- `data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Evidence_Assessment/paperinfo/paperinfo_PICOdff23ac6_c844628.json`
- `data/2021ACR RA/Recommendation_Formation/quicker_data(PICO_IDXdff23ac6)_20260601204748.json`
- `reports/run_manifest_dff23ac6.json`

## 7. 验证记录

- `config/config.json` 已通过 JSON 校验。
- `config/config-template.json` 已通过 JSON 校验。
- `main.py` 已通过 Python 字节码编译检查。
- 集成运行已到达 Phase5，并保存最终推荐结果。
- 停止请求后，未发现仍在运行的 `python main.py` 进程。
