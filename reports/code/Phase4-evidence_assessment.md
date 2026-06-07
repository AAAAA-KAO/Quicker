# Phase4-evidence_assessment.py 使用说明

该脚本由 `Phase4-evidence_assessment.ipynb` 整理而来，用于运行 Phase4 证据评价。它会读取 Phase1 的 PICO 信息、Phase3 的 `paperinfo/outcomeinfo`，调用 `Quicker.assess_evidence()` 完成 GRADE 相关证据评价，并将评价结果保存到 Evidence_Assessment 目录。

## 运行示例

```bash
python Phase4-evidence_assessment.py \
  --YOUR_CONFIG_PATH config/config.json \
  --YOUR_DATASET_PATH "data/2021ACR RA" \
  --YOUR_QUESTION_DECOMPOSITION_PATH "data/2021ACR RA/Question_Decomposition" \
  --YOUR_STUDY_SELECTION_PATH "data/2021ACR RA/Study_Selection" \
  --YOUR_EVIDENCE_ASSESSMENT_PATH "data/2021ACR RA/Evidence_Assessment" \
  --YOUR_PAPER_LIBRARY_PATH "data/2021ACR RA/Paper_Library" \
  --disease "Rheumatoid Arthritis (RA)" \
  --pico_idx dff23ac6
```

默认会先把 Phase3 的 `paperinfo` 和 `outcomeinfo` 中匹配 `pico_idx` 的文件复制到 Evidence_Assessment 目录，再执行证据评价。

## 输入参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--YOUR_CONFIG_PATH` | `config/config.json` | 项目配置文件。脚本会读取模型配置、embedding 配置和 `pipeline.phase4_evidence_assessment` 中的默认设置。 |
| `--YOUR_DATASET_PATH` | `data/2021ACR RA` | 数据集根目录，主要用于日志路径。 |
| `--YOUR_QUESTION_DECOMPOSITION_PATH` | `data/2021ACR RA/Question_Decomposition` | Phase1 输出目录，必须包含 `PICO_Information.json`。 |
| `--YOUR_LITERATURE_SEARCH_PATH` | `data/2021ACR RA/Literature_Search` | Literature_Search 目录，传给 `Quicker` 初始化。 |
| `--YOUR_STUDY_SELECTION_PATH` | `data/2021ACR RA/Study_Selection` | Phase3 输出目录。默认从这里复制 `paperinfo/outcomeinfo`。 |
| `--YOUR_EVIDENCE_ASSESSMENT_PATH` | `data/2021ACR RA/Evidence_Assessment` | Phase4 工作目录和输出目录。 |
| `--YOUR_PAPER_LIBRARY_PATH` | `data/2021ACR RA/Paper_Library` | 论文 PDF、解析结果和向量数据库所在目录。 |
| `--disease` | `Rheumatoid Arthritis (RA)` | 疾病或主题名称。 |
| `--pico_idx` | `dff23ac6` | 要评价的 PICO index，需存在于 `PICO_Information.json`。 |
| `--comparator` | 无 | 只运行指定 comparator；不传则运行该 PICO 下所有 comparator。 |
| `--input_comparator_postfix` | 无 | 手动指定输入文件后缀，例如 `_c649f30`。 |
| `--input_comparator_postfix_map_json` | 无 | comparator 到输入后缀的 JSON 对象或 JSON 文件路径，例如 `'{"Comparator text":"_c649f30"}'`。不传时优先使用配置文件里的 `input_comparator_postfix_map`。 |
| `--output_comparator_postfix_map_json` | 无 | comparator 到输出后缀的 JSON 对象或 JSON 文件路径。不传时使用配置文件里的 `output_comparator_postfix_map`；传 `'{}'` 可模拟 notebook 中 `quicker.comparator_postfix_map = {}` 的行为，输出不带 comparator 后缀。 |
| `--annotation_json` | 无 | evidence assessment annotation 的 JSON 对象或文件路径。不传时使用配置文件里的 `annotation`。 |
| `--transfer_study_selection_files` / `--no-transfer_study_selection_files` | 开启 | 是否先把 Phase3 的 `paperinfo/outcomeinfo` 复制到 Evidence_Assessment。 |
| `--derive_comparator_postfix` / `--no-derive_comparator_postfix` | 开启 | 是否根据 comparator 文本自动推导 `_c{md5前6位}` 后缀。 |
| `--skip_comparators_without_inputs` / `--no-skip_comparators_without_inputs` | 开启 | 找不到某个 comparator 的输入文件时，跳过还是报错。 |
| `--reuse_existing_outputs` | 关闭 | 如果 Evidence_Assessment 下已经存在已评价的 outcomeinfo，则跳过本次评价。 |
| `--print_state` | 关闭 | 每个 comparator 评价前打印 `QuickerData` 状态。 |

## 输入文件

### 1. 配置文件

路径由 `--YOUR_CONFIG_PATH` 指定，默认：

```text
config/config.json
```

至少需要包含：

- `model.evidence_assessment_model`
- `model.embeddings`
- `evidence_assessment.additional_requirements`
- 可选的 `pipeline.phase4_evidence_assessment`

### 2. PICO 信息

路径：

```text
{YOUR_QUESTION_DECOMPOSITION_PATH}/PICO_Information.json
```

脚本会用 `--pico_idx` 找到对应记录，并读取：

- `Question`
- `P`
- `I`
- `C`
- `O`

### 3. Phase3 研究筛选输出

默认从：

```text
{YOUR_STUDY_SELECTION_PATH}/paperinfo/
{YOUR_STUDY_SELECTION_PATH}/outcomeinfo/
```

复制到：

```text
{YOUR_EVIDENCE_ASSESSMENT_PATH}/paperinfo/
{YOUR_EVIDENCE_ASSESSMENT_PATH}/outcomeinfo/
```

每个 comparator 需要至少一组匹配文件：

```text
paperinfo_PICO{pico_idx}{postfix}.json
outcomeinfo_PICO{pico_idx}{postfix}.json
```

例如当前数据：

```text
paperinfo_PICOdff23ac6_c649f30.json
outcomeinfo_PICOdff23ac6_c649f30.json
```

`postfix` 的查找顺序：

1. `--input_comparator_postfix`
2. `--input_comparator_postfix_map_json`
3. 配置文件 `pipeline.phase4_evidence_assessment.input_comparator_postfix_map`
4. 自动推导 `_c{md5(comparator)前6位}`
5. 如果只有一个 comparator，自动发现 Evidence_Assessment 中成对存在的 `paperinfo/outcomeinfo` 后缀

### 4. 论文全文和向量库

路径：

```text
{YOUR_PAPER_LIBRARY_PATH}/PICO{pico_idx}/
```

`Quicker.assess_evidence()` 会调用 `paper.get_pdf()`。如果已有 PDF、TEI 或向量数据库，会复用；如果没有，可能触发下载、解析或向量库构建，具体取决于项目配置和本地环境。

## 输出文件

### 1. 证据评价 outcomeinfo

主要输出目录：

```text
{YOUR_EVIDENCE_ASSESSMENT_PATH}/outcomeinfo/
```

输出文件名通常为：

```text
outcomeinfo_PICO{pico_idx}{postfix}.json
```

其中 `{postfix}` 默认由 comparator 自动生成，例如：

```text
data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICOdff23ac6_c649f30.json
```

文件中每个 outcome 的 `assessment_results.GRADE` 会被补充证据评价结果，例如 risk of bias、原始数据抽取、certainty 等字段，具体取决于 `config.json` 中的 `evidence_assessment.additional_requirements`。

### 2. 可能更新的 paperinfo

如果评价过程中 paper 对象发生变化，脚本会保存：

```text
{YOUR_EVIDENCE_ASSESSMENT_PATH}/paperinfo/paperinfo_PICO{pico_idx}{postfix}.json
```

常见变化包括 PDF 获取状态、解析结果、研究特征或向量库相关状态。

### 3. Paper_Library 中的中间文件

证据评价可能在以下目录创建或更新文件：

```text
{YOUR_PAPER_LIBRARY_PATH}/PICO{pico_idx}/{paper_uid}/
```

可能包括：

- PDF 文件
- GROBID/TEI 解析结果
- 文本切块或元数据
- `{paper_uid}_vector_database/`

### 4. 日志文件

日志默认写入：

```text
log/{dataset_name}/Evidence_Assessment/{pico_idx}.log
```

其中 `{dataset_name}` 来自 `--YOUR_DATASET_PATH` 的最后一级目录。

## 与 notebook 的差异

- notebook 中硬编码了 `quicker.load_outcome_list(comparator_postfix="_c844628")` 和 `quicker.load_paper_list(comparator_postfix="_c844628")`；脚本改为自动查找输入后缀，并支持手动覆盖。
- notebook 中 `transfer_outcome_and_paperinfo(...)` 是注释示例；脚本默认执行该复制步骤，可用 `--no-transfer_study_selection_files` 关闭。
- notebook 中 `quicker.comparator_postfix_map = {}` 会让输出文件不带 comparator 后缀；脚本默认按 comparator 生成后缀。若要保持 notebook 行为，可传：

```bash
python Phase4-evidence_assessment.py --output_comparator_postfix_map_json '{}'
```
