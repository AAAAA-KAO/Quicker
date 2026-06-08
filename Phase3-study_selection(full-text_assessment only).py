"""
Phase3 第二阶段兼容入口。

运行示例：
    conda run -n quicker python "Phase3-study_selection(full-text_assessment only).py" \
        --YOUR_CONFIG_PATH config/config.json

脚本功能：
    该文件仅为兼容旧文件名保留。实际全文评估逻辑已拆分到
    Phase3-full_text_assessment.py。所有命令行参数会原样转发给新脚本。

输入输出与命令行参数：
    请参见 Phase3-full_text_assessment.py 顶部中文文档字符串。
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("Phase3-full_text_assessment.py")),
        run_name="__main__",
    )
