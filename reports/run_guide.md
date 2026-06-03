# Quicker 运行指南

## 前置条件

```bash
# 1. 复制并编辑配置文件
cp config/config-template.json config/config.json
# 填写 API_KEY, BASE_URL, model_name, clinical_question 等字段

# 2. 安装依赖
pip install -r requirements.txt
```

## 运行命令

```bash
python main.py --config config/config.json
```

## 各阶段输出文件

| 阶段                        | 说明                     | 输出文件                                                                                                                                                                                     |
| --------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase1 问题分解             | 将临床问题转为 PICO 结构 | `data/{dataset}/Question_Decomposition/PICO_Information.json`                                                                                                                              |
| Phase2 文献检索             | PubMed 检索并过滤        | `data/{dataset}/quicker_data(PICO_IDX{idx})_ls.json`                                                                                                                                       |
| Phase3 研究筛选             | 题录筛选 + 全文评估      | `data/{dataset}/Study_Selection/paperinfo/*.json<br>``data/{dataset}/Study_Selection/outcomeinfo/*.json<br>``data/{dataset}/Study_Selection/Results/screening_records/{method}/{idx}/` |
| Phase4 证据评价             | GRADE 证据质量评估       | `data/{dataset}/Evidence_Assessment/outcomeinfo/*.json`                                                                                                                                    |
| Phase4 Meta-Analysis (可选) | R meta 荟萃分析          | `reports/meta_analysis/{pico_idx}/*.txt`                                                                                                                                                   |
| Phase5 推荐形成             | 综合证据生成推荐         | `data/{dataset}/Recommendation_Formation/quicker_data(PICO_IDX{idx})_{timestamp}.json`                                                                                                     |

每次运行后还会在 `reports/` 下生成：

- `run_manifest_{pico_idx}.json` — 运行清单
- `experiment_report_{pico_idx}.md` — 实验报告

## 常用配置说明

- `pico_idx`: 设为 `"auto"` 自动根据临床问题生成，或手动指定
- `stages`: 按需开关各阶段 (`true`/`false`)
- 各阶段的 `reuse_existing_*`: 控制是否跳过已有输出
- `pdf_handling.stop_when_missing_pdf`: PDF 缺失时是否中断
