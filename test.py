import os
import pandas as pd
import json


def print_dict_pretty(dictionary: dict, indent: int = 4) -> None:
    """
    美观打印 Python 字典
    
    Args:
        dictionary: 要打印的字典
        indent: 缩进空格数，默认为 4
    """
    print(json.dumps(dictionary, indent=indent, ensure_ascii=False))


YOUR_QUESTION_DECOMPOSITION_PATH="data/2021ACR RA/Question_Decomposition"
pico_idx="ef0e4f95"

question_deconstruction_datapath = os.path.join(
YOUR_QUESTION_DECOMPOSITION_PATH, 'PICO_Information.json'
)
question_deconstruction_data = pd.read_json(
    question_deconstruction_datapath, dtype={'Index': str}
)

question_deconstruction_data = question_deconstruction_data[
    question_deconstruction_data['Index'] == pico_idx
]

original_qd_dict = question_deconstruction_data.to_dict(orient='records')
print_dict_pretty(original_qd_dict[0])
