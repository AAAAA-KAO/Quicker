# 1 Question Decomposition

输入：临床问题

输出：data/2021ACR RA/Question_Decomposition/PICO_Information.json

# 2 Literature Search

输入：

1. pico_idx：临床问题对应的索引

2. data/2021ACR RA/Question_Decomposition/PICO_Information.json

输出：

1. 检索策略：data/2021ACR RA/Literature_Search/pubmed/Results/{model_name}/use_agent_{True/False}/检索策略.txt
2. 检索结果：data/2021ACR RA/Literature_Search/pubmed/Results/{model_name}/use_agent_{True/False}/PICO{pico_idx}.json
3. 启发式筛选后的最终结果（含PICO和检索结果）：data/2021ACR RA/quicker_data(PICO_IDX{pico_idx})_ls.json

注意：执行这段代码需要关闭vpn

# 3 Study Selection

输入：

1. pico_idx：临床问题对应的索引
2. 阶段二生成的文献检索结果：data/2021ACR RA/quicker_data(PICO_IDX{pico_idx})_ls.json

输出：
