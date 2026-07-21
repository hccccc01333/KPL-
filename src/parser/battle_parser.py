"""
比赛级 JSON 解析器。

由 notebooks/03_数据清洗与建表.ipynb 调试后迁移。
"""
# TODO：
# def parse_battle(raw: dict) -> dict:
#     """从原始 JSON 提取比赛级字段，返回一行记录"""
#     ...
import json
import pandas as pd

def parse_battle(raw: dict):
    d = raw['data']
    c1,c2 = d['camp1'],d['camp2']

    col_list = []

    for c in d.keys():
        col_list.append(c)
    del col_list[5:10]

    results = {}
    for c in col_list:
        results[c] = d[c]
    for c in c1.keys():
        results[f"camp1_{c}"] = c1[c]
    for c in c2.keys():
        results[f"camp2_{c}"] = c2[c]
    return results
