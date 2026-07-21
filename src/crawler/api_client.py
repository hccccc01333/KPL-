"""
KPL 接口客户端。

封装单场比赛拉取，带重试 / 缓存 / 日志。
本文件代码由 notebooks/01_单场爬虫.ipynb 调试通过后迁移而来。
"""

import json
import time
import random
import requests

from src.config import (
    BASE_URL_BATTLE,
    HEADERS,
    RAW_DIR,
    TIMEOUT,
    MAX_RETRY,
    SLEEP_RANGE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_battle(
    battle_id: str,
    use_cache: bool = True
) -> dict | None:
    """
    拉取单场比赛原始 JSON 数据。

    Args:
        battle_id:
            对局号，例如 "1038107152_18_1742644777"

        use_cache:
            是否使用本地缓存

    Returns:
        成功返回 dict，失败返回 None
    """

    cache_path = RAW_DIR / f"{battle_id}.json"
    if not battle_id :
        logger.warning(f"{battle_id}为空")
        return None

    # 缓存命中
    if use_cache and cache_path.exists():
        logger.info(f"[命中缓存，{battle_id}]")

        with open(cache_path, encoding="utf-8-sig") as f:
            return json.load(f)

    # 网络请求 + 重试
    for i in range(MAX_RETRY):

        try:
            url = f"{BASE_URL_BATTLE}?battle_id={battle_id}"

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT
            )

            # HTTP 状态检查
            response.raise_for_status()

            data = response.json()

            # 业务错误
            if data.get("code") != 200:
                logger.warning(
                    f"{battle_id} 业务错误："
                    f"{data.get('code')}"
                )
                return None

            # 空数据
            if data.get("data") is None:
                logger.warning(
                    f"{battle_id} 返回空数据"
                )
                return None

            logger.info("已查询到数据，正在写入数据")

            # 保存缓存
            with open(
                cache_path,
                "w",
                encoding="utf-8-sig"
            ) as f:

                json.dump(
                    data,
                    f,
                    ensure_ascii=False,
                    indent=4
                )

            # 反爬 sleep
            time.sleep(
                random.uniform(*SLEEP_RANGE)
            )

            logger.info(
                f"数据保存成功，路径：{cache_path}"
            )

            return data

        # 可重试错误
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ) as e:

            wait_time = (
                2 ** i
                + random.uniform(0, 1)
            )

            logger.warning(
                f"第{i+1}次对局 "
                f"{battle_id} 提取失败，"
                f"{wait_time:.1f} 秒后重试。"
                f"错误信息：{e}"
            )

            time.sleep(wait_time)

        # 不可重试错误
        except requests.exceptions.RequestException as e:

            logger.warning(
                f"{battle_id} 请求失败：{e}"
            )

            return None

    logger.error(
        f"{battle_id} 已重试 "
        f"{MAX_RETRY} 次，均失败"
    )

    return None