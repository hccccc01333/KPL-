"""
项目全局配置文件。

所有 URL、Headers、路径、超参数集中在这里管理。
后续 src 下的模块都从这里 import 常量，避免硬编码散落各处。
"""
# ============================================================
# TODO（你完成 notebook 00 探索后填写）
# ============================================================

# --- 接口地址 ---
# 提示：从 notebook 00 步骤 2 找到的 URL，去掉 ?battle_id=xxx 那段
BASE_URL_BATTLE = "https://prod.comp.smoba.qq.com/leaguesite/battle/open"  # 单场比赛接口
BASE_URL_LEAGUE = "https://prod.comp.smoba.qq.com/leaguesite/matches/open"  # 赛程接口（所有 match）
BASE_URL_MATCH = "https://prod.comp.smoba.qq.com/leaguesite/match/battles/open"  # 单场 match 内所有 battle

# --- 请求头 ---
# 提示：从 scripts/01api.py 已经验证过的那一套
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://pvp.qq.com/",
    "Origin": "https://pvp.qq.com",
}

# --- 路径 ---
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

for _d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- 爬虫超参数 ---
TIMEOUT = 15
MAX_RETRY = 3
SLEEP_RANGE = (1.0, 2.0)  # 反爬延时范围（秒）

# --- 赛事 ID ---
LEAGUE_IDS = {
    "2026KPL夏季赛": 20260003,
    "2026挑战者杯": 20260002,
}

# --- 比赛ID ---
MATCH_IDS = {

}