"""
赛程爬虫：从 league_id 出发拉取整个赛季的所有 match 和 battle_id。

本文件代码由 notebooks/02_批量爬取.ipynb 调试通过后迁移而来。
"""

import requests
import pandas as pd
from tqdm import tqdm

from src.config import (
    BASE_URL_LEAGUE,
    BASE_URL_MATCH,
    HEADERS,
    PROCESSED_DIR,
    TIMEOUT,
)
from src.crawler.api_client import fetch_battle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_schedule(league_id: int) -> pd.DataFrame:
    """
    拉取某赛季的完整赛程表。

    Args:
        league_id: 赛事 ID，例如 20260002（2026挑战者杯）

    Returns:
        赛程 DataFrame（一行 = 一场 match）
    """
    url = f"{BASE_URL_LEAGUE}?league_id={league_id}"
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()

    schedule_df = pd.DataFrame(data["results"])
    logger.info(f"拉取赛程成功：{len(schedule_df)} 场 match")

    schedule_df.to_csv(
        PROCESSED_DIR / "schedule.csv",
        index=False,
        encoding="utf-8-sig",
    )
    logger.info(f"赛程已保存至 {PROCESSED_DIR / 'schedule.csv'}")

    return schedule_df


def fetch_match_battles(match_id: str) -> list[str]:
    """
    拉取某场 match 下的所有 battle_id。

    Args:
        match_id: 比赛 ID，例如 "2026042501"

    Returns:
        battle_id 列表
    """
    url = f"{BASE_URL_MATCH}?match_id={match_id}"
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()

    return [r["battle_id"] for r in data.get("results", [])]


def batch_crawl(league_id: int) -> tuple[list[str], list[tuple[str, str]]]:
    """
    全赛季批量爬取：赛程 → match_id → battle_id → fetch_battle。

    Args:
        league_id: 赛事 ID

    Returns:
        (success_ids, failed_ids) 元组
    """
    schedule_df = fetch_schedule(league_id)

    completed = schedule_df[schedule_df["status"] == 2]#已经完赛的
    match_ids = completed["match_id"].unique().tolist()
    logger.info(f"已完赛 {len(completed)} 场 match，开始拉取 battle_id")

    all_battle_ids = []
    for mid in match_ids:
        all_battle_ids.extend(fetch_match_battles(mid))

    logger.info(f"共 {len(all_battle_ids)} 个 battle_id，开始批量爬取")

    success, failed = [], []
    for bid in tqdm(all_battle_ids, desc="批量爬取"):
        try:
            data = fetch_battle(bid)
            if data is not None:
                success.append(bid)
            else:
                failed.append((bid, "fetch returned None"))
        except Exception as e:
            failed.append((bid, str(e)))

    logger.info(
        f"爬取完成：成功 {len(success)}，失败 {len(failed)}，"
        f"成功率 {len(success)/max(len(all_battle_ids),1):.1%}"
    )

    return success, failed
