"""选手级 JSON 解析器。由 notebooks/03 调试后迁移。"""
# TODO：
# def parse_players(raw: dict) -> list[dict]:
#     ...
import pandas as pd

def parse_players(raw: dict):

    df = pd.json_normalize(raw['data']['battle_player_list'])
    df['battle_id'] = raw['data']['battle_id']
    return df

