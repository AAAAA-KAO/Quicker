# Phase3-Full Text、Phase4、Phase5 运行前服务与配置报告

本文档分析以下三个脚本的运行前置条件、是否需要额外服务、输入输出依赖和完整配置项：

- `Phase3-full_text_assessment.py`
- `Phase4-evidence_assessment.py`
- `Phase5-recommendation_formulation.py`

当前项目配置文件为 `config/config.json`。以下示例基于当前配置中的数据集：

- `dataset_name`: `2021ACR RA`
- `pico_idx`: `auto`，按当前问题会解析为 `dff23ac6`
- `dataset_path`: `data/2021ACR RA`
- `paper_library`: `data/2021ACR RA/Paper_Library`

注意：报告中的模型 API Key 以 `<REDACTED>` 表示，实际运行仍读取本地 `config/config.json`。

## 总览

| 脚本 | 需要启动 GROBID | 需要启动 Qdrant Server | 需要模型 API | 需要 Embedding API | 需要本地 PDF | 主要输入 |
| --- | --- | --- | --- | --- | --- | --- |
| `Phase3-full_text_assessment.py` | 是，除非所有论文已完成全文评估且不重跑抽取 | 否 | 是，`study_selection_model` | 是，`embeddings` | 是 | Phase3 题录纳入 JSON、Phase2 quicker data、PICO、PDF |
| `Phase4-evidence_assessment.py` | 视情况而定；若向量库缺失或失效则需要 | 否 | 是，`evidence_assessment_model` | 是，`embeddings` | 是 | Phase3 `paperinfo/outcomeinfo`、PICO、PDF/向量库 |
| `Phase5-recommendation_formulation.py` | 否 | 否 | 是，`recommendation_formation_model` | 否 | 否 | Phase4 `outcomeinfo`、PICO |

结论：

- GROBID 是 Phase3 全文评估的关键外部服务，也可能在 Phase4 重建向量库时被用到。
- Qdrant 不需要以 server 方式启动；代码使用的是本地嵌入式 `QdrantClient(path=...)`。
- Phase3/Phase4 的向量库保存在每篇论文目录下，例如 `data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/{paper_uid}_vector_database`。
- Phase5 不需要 PDF、GROBID、Qdrant 或 embeddings，只需要 Phase4 证据评价结果和推荐形成模型。

## 外部服务与本地组件

### 1. GROBID

需要启动。代码位置：

- `utils/Evidence_Assessment/PDFprocessing.py`
- `CustomizedGrobidParser(..., grobid_server="http://localhost:8070/api/processFulltextDocument")`

用途：

- 从 PDF 中抽取正文、章节、句子、坐标、摘要等结构化文本。
- `Paper.extract_text_from_pdf()` 会调用 `CustomizedGrobidParser`。
- `Paper.get_vector_store()` 在创建向量库时会调用 `extract_text_from_pdf()`。

默认服务地址：

```text
http://localhost:8070/api/processFulltextDocument
```

运行前检查：

```bash
curl http://localhost:8070/api/isalive
```

常见 Docker 启动示例：

```bash
docker run --rm --name grobid -p 8070:8070 lfoppiano/grobid:0.8.0
```

如果你使用本地安装或已有 Docker 镜像，只要最终暴露 `localhost:8070` 即可。

当前限制：

- GROBID URL 没有放在 `config/config.json` 中。
- 若要改端口或远程地址，需要修改 `CustomizedGrobidParser` 的默认参数，或后续给脚本增加配置项。

### 2. Qdrant

不需要启动独立 Qdrant Server。

代码使用方式：

```python
QdrantClient(path=self.vectorstore_save_path)
```

这表示 Qdrant 以本地文件方式运行，向量库直接写入论文目录。不是访问 `localhost:6333` 之类的 Qdrant 服务。

向量库路径示例：

```text
data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/{paper_uid}_vector_database
```

注意事项：

- 不要并发运行多个进程同时写同一篇论文的同一个 Qdrant 本地目录，可能产生文件锁或损坏。
- 如果向量库目录已存在但 collection 缺失、为空或无法读取，代码会删除并重建该目录。
- 向量维度在 `Paper.get_vector_store()` 中硬编码为 `1024`，需要与当前 embedding 模型输出维度一致。

### 3. Unstructured 表格抽取

不需要启动单独服务，但需要本地 Python 依赖和可能的系统依赖。

代码位置：

- `Paper.extract_table_from_pdf()`
- 使用 `langchain_unstructured.UnstructuredLoader(strategy="hi_res", skip_infer_table_types=[])`

用途：

- 在创建向量库时抽取 PDF 表格。
- 表格抽取后会调用当前阶段的大模型，把表格 HTML/内容描述成文本，再加入 Qdrant 向量库。

注意事项：

- `strategy="hi_res"` 通常比普通文本抽取依赖更多本地组件和模型。
- 代码设置了 `HF_ENDPOINT=https://hf-mirror.com`，首次运行或模型未缓存时可能访问 Hugging Face 镜像。
- 如果本地 unstructured 高精度 PDF 解析依赖不完整，Phase3/Phase4 的向量库创建可能失败。

### 4. 模型服务与网络

三个脚本都需要联网访问模型服务，除非你把模型配置改成本地兼容服务。

当前配置使用 OpenAI-compatible 接口：

```json
{
  "provider": "OpenAI",
  "model_name": "qwen-plus",
  "API_KEY": "<REDACTED>",
  "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "temperature": 1.0
}
```

Phase3 使用：

- `model.study_selection_model`
- `model.embeddings`

Phase4 使用：

- `model.evidence_assessment_model`
- `model.embeddings`

Phase5 使用：

- `model.recommendation_formation_model`

Embedding 当前配置：

```json
{
  "provider": "OpenAI",
  "model_name": "text-embedding-v4",
  "API_KEY": "<REDACTED>",
  "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "kwargs": {
    "check_embedding_ctx_length": false
  }
}
```

重要：`Paper.get_vector_store()` 中 Qdrant collection 维度硬编码为 `1024`。如果换 embedding 模型，需要确认输出维度仍为 1024，或修改代码中的 `VectorParams(size=1024, ...)`。

## Phase3-full_text_assessment.py

### 功能

第二阶段全文评估。它读取 Phase3 第一阶段生成的题录纳入论文列表，检查 PDF 是否齐全，然后对每篇论文做全文 RAG 分析，抽取：

- 研究设计
- population
- intervention
- comparator
- outcome

随后根据 comparator 匹配论文，生成：

- `paperinfo`
- `outcomeinfo`
- full text assessment 的 QuickerData 运行结果

### 是否需要额外服务

需要：

- GROBID server：默认 `localhost:8070`
- 模型服务：`study_selection_model`
- Embedding 服务：`embeddings`

不需要：

- Qdrant server

可能需要：

- unstructured hi-res PDF 解析依赖
- Hugging Face 模型下载或缓存，取决于本机 unstructured 配置

### PDF 行为

脚本中显式设置：

```python
os.environ["QUICKER_DISABLE_PDF_DOWNLOAD"] = "1"
```

因此它不会自动下载 PDF。缺 PDF 时：

- 先生成缺失 PDF 清单。
- 如果 `stop_when_missing_pdf=true`，脚本停止。
- 如果强行 `--no-stop_when_missing_pdf`，后续 `paper.get_pdf()` 仍会因为禁用下载而失败。

当前配置中：

```json
"pdf_handling": {
  "stop_when_missing_pdf": true,
  "missing_pdf_json": "missing_pdfs_{stage}_PICO{pico_idx}.json",
  "missing_pdf_markdown": "missing_pdfs_{stage}_PICO{pico_idx}.md"
}
```

所以 `Phase3-full_text_assessment.py` 当前默认仍会按配置生成 JSON 和 Markdown 缺失 PDF 清单到 `reports` 目录。若不希望生成 Markdown，可运行时覆盖：

```bash
conda run -n quicker python Phase3-full_text_assessment.py \
  --YOUR_CONFIG_PATH config/config.json \
  --missing_pdf_markdown_path ""
```

### 运行前必须准备

1. Phase1 输出：

```text
data/2021ACR RA/Question_Decomposition/PICO_Information.json
```

2. Phase2 输出：

```text
data/2021ACR RA/quicker_data(PICO_IDXdff23ac6)_ls.json
```

注意：当前脚本默认查找的是：

```text
data/2021ACR RA/quicker_data(PICO_IDXdff23ac6)_ls.json
```

这里 `PICO_IDX{pico_idx}` 和此前 Phase2 实际输出 `PICO_IDXdff23ac6` 是一致的字符串拼接结果。

3. Phase3 第一阶段题录纳入 JSON：

```text
data/2021ACR RA/Study_Selection/record_included_studies/record_included_PICOdff23ac6.json
```

4. 每篇纳入论文的 PDF：

```text
data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/*.pdf
```

### 主要配置字段

路径配置：

```json
"pipeline": {
  "paths": {
    "dataset": "data/2021ACR RA",
    "question_decomposition": "data/2021ACR RA/Question_Decomposition",
    "literature_search": "data/2021ACR RA/Literature_Search",
    "study_selection": "data/2021ACR RA/Study_Selection",
    "paper_library": "data/2021ACR RA/Paper_Library",
    "reports": "reports"
  }
}
```

全文评估配置：

```json
"study_selection": {
  "full_text_assessment_method": "RAG",
  "reupdate_component_list": [
    "population",
    "intervention",
    "comparator",
    "outcome"
  ]
}
```

Phase3 业务配置：

```json
"pipeline": {
  "phase3_study_selection": {
    "study": ["randomized clinical trial"],
    "inclusion_criteria": "",
    "exclusion_criteria": ""
  }
}
```

模型配置：

```json
"model": {
  "study_selection_model": {
    "provider": "OpenAI",
    "model_name": "qwen-plus",
    "API_KEY": "<REDACTED>",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 1.0
  },
  "embeddings": {
    "provider": "OpenAI",
    "model_name": "text-embedding-v4",
    "API_KEY": "<REDACTED>",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kwargs": {
      "check_embedding_ctx_length": false
    }
  }
}
```

运行环境：

```json
"pipeline": {
  "runtime_environment": {
    "TOP_K": "5"
  }
}
```

`TOP_K` 会被 `prepare_environment()` 写入环境变量，并在 RAG 提取 PICO 组件时控制相似检索数量。

### 推荐运行命令

```bash
conda run -n quicker python Phase3-full_text_assessment.py \
  --YOUR_CONFIG_PATH config/config.json
```

如果只想生成 JSON 缺失 PDF 清单，不生成 Markdown：

```bash
conda run -n quicker python Phase3-full_text_assessment.py \
  --YOUR_CONFIG_PATH config/config.json \
  --missing_pdf_markdown_path ""
```

## Phase4-evidence_assessment.py

### 功能

证据评价。它读取 Phase1 的 PICO 信息和 Phase3 全文评估输出的 `paperinfo/outcomeinfo`，按 comparator 加载证据，调用 `Quicker.assess_evidence()` 完成 GRADE 相关评价。

当前配置下重点执行：

- GRADE `risk of bias`
- `extract_raw_data=true`
- RCT 风险偏倚方法：`quicker`

### 是否需要额外服务

需要：

- 模型服务：`evidence_assessment_model`
- Embedding 服务：`embeddings`

视情况需要：

- GROBID server：如果 Phase3 已经生成可用向量库，则通常不需要；如果 Phase4 发现向量库缺失、为空、损坏或需要重建，则需要。
- unstructured hi-res PDF 解析依赖：同样只在重建向量库时需要。

不需要：

- Qdrant server。仍然使用本地 `QdrantClient(path=...)`。

### PDF 行为

Phase4 会对每个 `paper_list` 调用：

```python
paper.get_pdf(current_save_folder=self.paper_library_path)
```

当前 Phase4 不会主动设置 `QUICKER_DISABLE_PDF_DOWNLOAD=1`。但是 `Paper.download_pdf()` 中真正联网下载的 PyPaperBot 代码目前是注释状态；实际逻辑主要是：

- 若 PDF 已在 `Paper_Library/PICO{pico_idx}/{paper_uid}` 下，直接使用。
- 若设置了环境变量 `PAPER_SOURCE_DIR`，尝试从 `PAPER_SOURCE_DIR/{paper_uid}/*.pdf` 拷贝到 Paper_Library。
- 否则找不到 PDF 时失败。

因此 Phase4 不依赖自动网络下载 PDF，但必须能找到本地 PDF 或本地 PDF 源。

### 运行前必须准备

1. Phase1 输出：

```text
data/2021ACR RA/Question_Decomposition/PICO_Information.json
```

2. Phase3 全文评估输出：

```text
data/2021ACR RA/Study_Selection/paperinfo/
data/2021ACR RA/Study_Selection/outcomeinfo/
```

3. 本地 PDF：

```text
data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/*.pdf
```

4. 可选但推荐：Phase3 已构建好的本地 Qdrant 向量库：

```text
data/2021ACR RA/Paper_Library/PICOdff23ac6/{paper_uid}/{paper_uid}_vector_database
```

如果向量库不存在，Phase4 会尝试创建，因此需要 GROBID 和 unstructured 环境就绪。

### 主要配置字段

路径配置：

```json
"pipeline": {
  "paths": {
    "question_decomposition": "data/2021ACR RA/Question_Decomposition",
    "literature_search": "data/2021ACR RA/Literature_Search",
    "study_selection": "data/2021ACR RA/Study_Selection",
    "evidence_assessment": "data/2021ACR RA/Evidence_Assessment",
    "paper_library": "data/2021ACR RA/Paper_Library"
  }
}
```

Phase4 行为配置：

```json
"pipeline": {
  "phase4_evidence_assessment": {
    "reuse_existing_outputs": true,
    "transfer_study_selection_files": true,
    "require_assessed_outputs_for_skip": true,
    "skip_when_dependency_missing": true,
    "input_comparator_postfix_map": {},
    "derive_comparator_postfix": true,
    "skip_comparators_without_inputs": true,
    "output_comparator_postfix_map": null,
    "annotation": {}
  }
}
```

证据评价配置：

```json
"evidence_assessment": {
  "additional_requirements": {
    "design_to_be_assessed": [
      "RANDOMIZED_CONTROLLED_TRIAL"
    ],
    "GRADE": {
      "rob_rcts": {
        "method": "quicker"
      },
      "factor_list": [
        "risk of bias"
      ],
      "extract_raw_data": true
    }
  }
}
```

模型配置：

```json
"model": {
  "evidence_assessment_model": {
    "provider": "OpenAI",
    "model_name": "qwen-plus",
    "API_KEY": "<REDACTED>",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 1.0
  },
  "embeddings": {
    "provider": "OpenAI",
    "model_name": "text-embedding-v4",
    "API_KEY": "<REDACTED>",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kwargs": {
      "check_embedding_ctx_length": false
    }
  }
}
```

### Comparator 后缀配置

Phase4 会按 comparator 后缀查找 Phase3 输出。默认逻辑：

- 若 `derive_comparator_postfix=true`，用 comparator 文本的 MD5 前 6 位生成后缀，例如 `_cxxxxxx`。
- 若只有一个 comparator，会额外尝试空后缀和自动发现。
- 可用 `input_comparator_postfix_map_json` 显式指定 comparator 到输入后缀的映射。

如果 Phase3 输出文件名与 Phase4 推断不一致，Phase4 会找不到 `paperinfo/outcomeinfo`。此时应配置：

```bash
--input_comparator_postfix_map_json '{"comparator text": "_c123abc"}'
```

或者在 `config.pipeline.phase4_evidence_assessment.input_comparator_postfix_map` 中配置。

### 推荐运行命令

```bash
conda run -n quicker python Phase4-evidence_assessment.py \
  --YOUR_CONFIG_PATH config/config.json
```

只评价某个 comparator：

```bash
conda run -n quicker python Phase4-evidence_assessment.py \
  --YOUR_CONFIG_PATH config/config.json \
  --comparator "continue DMARDs at the same doses"
```

## Phase5-recommendation_formulation.py

### 功能

推荐意见形成。它读取 Phase4 的 `outcomeinfo`，按 comparator 汇总每个 outcome 的证据评价结果，然后调用推荐形成模型生成：

- 每个 outcome 的解释
- 每个 comparator 的 summary
- rationale
- final recommendation

### 是否需要额外服务

需要：

- 模型服务：`recommendation_formation_model`

不需要：

- GROBID
- Qdrant server
- Embedding API
- PDF
- unstructured

Phase5 是三个脚本里服务依赖最少的一个。

### 运行前必须准备

1. Phase1 输出：

```text
data/2021ACR RA/Question_Decomposition/PICO_Information.json
```

2. Phase4 输出：

```text
data/2021ACR RA/Evidence_Assessment/outcomeinfo/outcomeinfo_PICO*.json
```

如果配置中：

```json
"transfer_evidence_assessment_files": true
```

脚本会先把 Phase4 的 `outcomeinfo/paperinfo` 复制到：

```text
data/2021ACR RA/Recommendation_Formation/
```

然后从 `Recommendation_Formation` 下读取 `outcomeinfo_PICO*.json`。

### 主要配置字段

路径配置：

```json
"pipeline": {
  "paths": {
    "question_decomposition": "data/2021ACR RA/Question_Decomposition",
    "evidence_assessment": "data/2021ACR RA/Evidence_Assessment",
    "recommendation_formation": "data/2021ACR RA/Recommendation_Formation"
  }
}
```

Phase5 行为配置：

```json
"pipeline": {
  "phase5_recommendation_formulation": {
    "reuse_existing_result": false,
    "transfer_evidence_assessment_files": true,
    "require_assessed_evidence": false,
    "overall_certainty": "LOW",
    "supplementary_information": ""
  }
}
```

模型配置：

```json
"model": {
  "recommendation_formation_model": {
    "provider": "OpenAI",
    "model_name": "qwen-plus",
    "API_KEY": "<REDACTED>",
    "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 1.0
  }
}
```

### 推荐运行命令

```bash
conda run -n quicker python Phase5-recommendation_formulation.py \
  --YOUR_CONFIG_PATH config/config.json
```

覆盖总体证据确定性：

```bash
conda run -n quicker python Phase5-recommendation_formulation.py \
  --YOUR_CONFIG_PATH config/config.json \
  --overall_certainty "LOW"
```

传入补充信息：

```bash
conda run -n quicker python Phase5-recommendation_formulation.py \
  --YOUR_CONFIG_PATH config/config.json \
  --supplementary_information "请在推荐中考虑患者偏好和长期用药负担。"
```

## 完整配置检查清单

运行 Phase3-full text、Phase4、Phase5 前，建议检查以下配置。

### 通用配置

```json
"logging": {
  "log_dir": "log",
  "log_file_name": "main_pipeline.log"
}
```

```json
"pipeline": {
  "dataset_name": "2021ACR RA",
  "dataset_path": "data/2021ACR RA",
  "disease": "Rheumatoid Arthritis (RA)",
  "clinical_question": "...",
  "pico_idx": "auto",
  "runtime_environment": {
    "TOP_K": "5"
  }
}
```

### 路径配置

```json
"pipeline": {
  "paths": {
    "dataset": "data/2021ACR RA",
    "question_decomposition": "data/2021ACR RA/Question_Decomposition",
    "literature_search": "data/2021ACR RA/Literature_Search",
    "study_selection": "data/2021ACR RA/Study_Selection",
    "evidence_assessment": "data/2021ACR RA/Evidence_Assessment",
    "paper_library": "data/2021ACR RA/Paper_Library",
    "recommendation_formation": "data/2021ACR RA/Recommendation_Formation",
    "reports": "reports"
  }
}
```

### PDF 配置

```json
"pipeline": {
  "pdf_handling": {
    "stop_when_missing_pdf": true,
    "local_pdf_source_dir": "papers/PICO{pico_idx}",
    "missing_pdf_json": "missing_pdfs_{stage}_PICO{pico_idx}.json",
    "missing_pdf_markdown": "missing_pdfs_{stage}_PICO{pico_idx}.md"
  }
}
```

说明：

- `stop_when_missing_pdf` 被 `Phase3-full_text_assessment.py` 使用。
- `missing_pdf_json` 和 `missing_pdf_markdown` 被 `Phase3-full_text_assessment.py` 使用。
- `local_pdf_source_dir` 在这三个脚本当前代码中没有被直接使用。
- Phase4 可通过环境变量 `PAPER_SOURCE_DIR` 从本地源目录复制 PDF，但这不是 `config/config.json` 中的配置项。

### 模型配置

Phase3：

```json
"study_selection_model": {
  "provider": "OpenAI",
  "model_name": "qwen-plus",
  "API_KEY": "<REDACTED>",
  "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "temperature": 1.0
}
```

Phase4：

```json
"evidence_assessment_model": {
  "provider": "OpenAI",
  "model_name": "qwen-plus",
  "API_KEY": "<REDACTED>",
  "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "temperature": 1.0
}
```

Phase5：

```json
"recommendation_formation_model": {
  "provider": "OpenAI",
  "model_name": "qwen-plus",
  "API_KEY": "<REDACTED>",
  "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "temperature": 1.0
}
```

Embedding：

```json
"embeddings": {
  "provider": "OpenAI",
  "model_name": "text-embedding-v4",
  "API_KEY": "<REDACTED>",
  "BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "kwargs": {
    "check_embedding_ctx_length": false
  }
}
```

### Phase3 配置

```json
"pipeline": {
  "phase3_study_selection": {
    "reuse_existing_outputs": true,
    "study": [
      "randomized clinical trial"
    ],
    "inclusion_criteria": "",
    "exclusion_criteria": ""
  }
}
```

```json
"study_selection": {
  "record_screening_method": "basic",
  "full_text_assessment_method": "RAG",
  "exp_num": 3,
  "threshold": 2,
  "reupdate_component_list": [
    "population",
    "intervention",
    "comparator",
    "outcome"
  ]
}
```

说明：

- `record_screening_method`、`exp_num`、`threshold` 主要属于 Phase3 第一阶段。
- `full_text_assessment_method` 和 `reupdate_component_list` 被 `Phase3-full_text_assessment.py` 使用。

### Phase4 配置

```json
"pipeline": {
  "phase4_evidence_assessment": {
    "reuse_existing_outputs": true,
    "transfer_study_selection_files": true,
    "input_comparator_postfix_map": {},
    "derive_comparator_postfix": true,
    "skip_comparators_without_inputs": true,
    "output_comparator_postfix_map": null,
    "annotation": {}
  }
}
```

```json
"evidence_assessment": {
  "additional_requirements": {
    "design_to_be_assessed": [
      "RANDOMIZED_CONTROLLED_TRIAL"
    ],
    "GRADE": {
      "rob_rcts": {
        "method": "quicker"
      },
      "factor_list": [
        "risk of bias"
      ],
      "extract_raw_data": true
    }
  }
}
```

### Phase5 配置

```json
"pipeline": {
  "phase5_recommendation_formulation": {
    "reuse_existing_result": false,
    "transfer_evidence_assessment_files": true,
    "require_assessed_evidence": false,
    "overall_certainty": "LOW",
    "supplementary_information": ""
  }
}
```

## 运行前 Preflight Checklist

### 服务检查

GROBID：

```bash
curl http://localhost:8070/api/isalive
```

Qdrant：

```text
不需要启动 qdrant server。
```

模型服务：

```text
确认 BASE_URL 可访问，API_KEY 有效，网络权限可用。
```

### 文件检查

PICO：

```bash
test -f "data/2021ACR RA/Question_Decomposition/PICO_Information.json"
```

Phase2 quicker data：

```bash
test -f "data/2021ACR RA/quicker_data(PICO_IDXdff23ac6)_ls.json"
```

Phase3 题录纳入：

```bash
test -f "data/2021ACR RA/Study_Selection/record_included_studies/record_included_PICOdff23ac6.json"
```

PDF：

```bash
find "data/2021ACR RA/Paper_Library/PICOdff23ac6" -name "*.pdf" | head
```

Phase3 full text 输出：

```bash
find "data/2021ACR RA/Study_Selection/paperinfo" -name "paperinfo_PICOdff23ac6*.json"
find "data/2021ACR RA/Study_Selection/outcomeinfo" -name "outcomeinfo_PICOdff23ac6*.json"
```

Phase4 输出：

```bash
find "data/2021ACR RA/Evidence_Assessment/outcomeinfo" -name "outcomeinfo_PICOdff23ac6*.json"
```

### Python 环境检查

```bash
conda run -n quicker python -m py_compile \
  Phase3-full_text_assessment.py \
  Phase4-evidence_assessment.py \
  Phase5-recommendation_formulation.py
```

关键 Python 包：

- `langchain_openai`
- `langchain_qdrant`
- `qdrant_client`
- `langchain_unstructured`
- `unstructured`
- `beautifulsoup4`
- `requests`
- `pandas`

## 当前代码中不在 config/config.json 管理的事项

以下内容目前是代码写死或通过环境变量控制，不属于 `config/config.json` 的完整配置范围：

| 项目 | 当前位置 | 当前值/行为 |
| --- | --- | --- |
| GROBID URL | `utils/Evidence_Assessment/PDFprocessing.py` | `http://localhost:8070/api/processFulltextDocument` |
| Qdrant 模式 | `utils/Evidence_Assessment/paper.py` | 本地 `QdrantClient(path=...)` |
| Qdrant 向量维度 | `utils/Evidence_Assessment/paper.py` | `1024` |
| Unstructured 策略 | `utils/Evidence_Assessment/paper.py` | `strategy="hi_res"` |
| HF 镜像 | `utils/Evidence_Assessment/paper.py` | `HF_ENDPOINT=https://hf-mirror.com` |
| Phase3 禁止自动 PDF 下载 | `Phase3-full_text_assessment.py` | `QUICKER_DISABLE_PDF_DOWNLOAD=1` |
| Phase4 本地 PDF 源 | 环境变量 | `PAPER_SOURCE_DIR`，仅在 `Paper.download_pdf()` 中读取 |

如果希望这些内容也可由 `config/config.json` 管理，需要进一步改造脚本和工具函数。

