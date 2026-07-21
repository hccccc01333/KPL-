"""
脚本 01：KPL 接口连通性测试
========================================
用途：验证腾讯 KPL 比赛数据接口是否可用。

✅ 验证记录：2026-05-05 跑通，状态码 200，返回 39187 字节合法 JSON。
   见 docs/接口逆向笔记.md。

运行方式：
    在 PyCharm 里右键 → Run '01_test_api'
    或命令行：python scripts/01api.py
"""
import requests
import json

# ============ 接口与 Headers ============
URL = "https://prod.comp.smoba.qq.com/leaguesite/battle/open"
TEST_BATTLE_ID = "1038107152_18_1742644777"  # 2024 春季赛 AG vs 重庆狼队

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Mobile Safari/537.36 Edg/128.0.0.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://pvp.qq.com/",
    "Origin": "https://pvp.qq.com",
}


def test_battle_api(battle_id: str) -> None:
    """测试单场比赛接口"""
    full_url = f"{URL}?battle_id={battle_id}"
    print(f"\n请求 URL: {full_url}")
    print("=" * 60)

    try:
        resp = requests.get(full_url, headers=HEADERS, timeout=15)
        print(f"状态码: {resp.status_code}")
        print(f"响应长度: {len(resp.text)} 字节")
        print(f"Content-Type: {resp.headers.get('Content-Type')}")

        if resp.status_code != 200:
            print(f"⚠️  状态码异常，前 500 字:\n{resp.text[:500]}")
            return

        data = json.loads(resp.text)
        print(f"\n业务码: {data.get('code')}, message: '{data.get('message')}'")
        print(f"顶层 keys: {list(data.keys())}")

        if "data" in data:
            inner = data["data"]
            print(f"\ndata 下二级 keys (共 {len(inner)} 个):")
            for k in inner.keys():
                v = inner[k]
                v_type = type(v).__name__
                v_preview = (
                    f"list of {len(v)}" if isinstance(v, list)
                    else f"dict with {len(v)} keys" if isinstance(v, dict)
                    else str(v)[:50]
                )
                print(f"  - {k:<25} ({v_type}): {v_preview}")

            # 战队基本信息
            for camp_key in ("camp1", "camp2"):
                if camp_key in inner:
                    camp = inner[camp_key]
                    print(
                        f"\n{camp_key}: {camp.get('team_name')} "
                        f"({camp.get('team_abbreviation')}) | "
                        f"胜负: {camp.get('is_win')} | "
                        f"KDA: {camp.get('kda')} | "
                        f"经济: {camp.get('gold')}"
                    )

    except requests.RequestException as e:
        print(f"❌ 网络异常: {type(e).__name__}: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("【测试】腾讯 KPL 比赛数据接口连通性")
    test_battle_api(TEST_BATTLE_ID)
