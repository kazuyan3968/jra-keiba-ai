import re
import time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

JRA_VENUE_CODE = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/",
}


def decode_html(resp: requests.Response) -> str:
    """レスポンスのHTMLヘッダやバイト列から文字コードを自動判別してデコードする"""
    raw = resp.content
    head = raw[:3000].lower()
    if b"utf-8" in head or b"utf8" in head:
        return raw.decode("utf-8", errors="replace")
    if b"euc-jp" in head or b"euc_jp" in head:
        return raw.decode("euc_jp", errors="replace")
    if b"shift_jis" in head or b"shift-jis" in head or b"cp932" in head:
        return raw.decode("cp932", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("euc_jp", errors="replace")


def fetch_race_list_by_date(date_str: str) -> list[dict]:
    """指定日（YYYY-MM-DD）のJRA開催レース一覧を取得する"""
    ymd = date_str.replace("-", "")
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={ymd}"

    time.sleep(1.2)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    html_text = decode_html(resp)

    soup = BeautifulSoup(html_text, "lxml")
    races = []
    seen_ids = set()

    for li in soup.select("li.RaceList_DataItem"):
        a_tag = li.find("a", href=True)
        if not a_tag:
            continue
        m = re.search(r"race_id=(\d{12})", a_tag["href"])
        if not m:
            continue
        rid = m.group(1)
        if rid in seen_ids:
            continue
        venue_code = rid[4:6]
        if venue_code not in JRA_VENUE_CODE:
            continue

        seen_ids.add(rid)
        venue_name = JRA_VENUE_CODE[venue_code]
        race_num = int(rid[10:12])

        title_el = li.select_one("span.ItemTitle")
        race_title = title_el.get_text(strip=True) if title_el else f"{race_num}R"

        time_el = li.select_one("span.RaceList_Itemtime")
        start_time = time_el.get_text(strip=True) if time_el else ""

        data_el = li.select_one("span.RaceList_ItemLong")
        course_info = data_el.get_text(strip=True) if data_el else ""

        label = f"{venue_name} {race_num:02d}R | {race_title} ({start_time} {course_info})".strip()
        races.append({
            "race_id": rid,
            "venue": venue_name,
            "race_num": race_num,
            "race_title": race_title,
            "start_time": start_time,
            "label": label,
        })

    if not races:
        db_url = f"https://db.netkeiba.com/race/list/{ymd}/"
        time.sleep(1.5)
        db_resp = requests.get(db_url, headers=HEADERS, timeout=15)
        db_soup = BeautifulSoup(decode_html(db_resp), "lxml")
        for a_tag in db_soup.select('a[href*="/race/"]'):
            m = re.search(r"/race/(\d{12})/", a_tag.get("href", ""))
            if not m:
                continue
            rid = m.group(1)
            venue_code = rid[4:6]
            if venue_code not in JRA_VENUE_CODE or rid in seen_ids:
                continue
            seen_ids.add(rid)
            venue_name = JRA_VENUE_CODE[venue_code]
            race_num = int(rid[10:12])
            r_title = a_tag.get_text(strip=True) or f"{race_num}R"
            races.append({
                "race_id": rid,
                "venue": venue_name,
                "race_num": race_num,
                "race_title": r_title,
                "start_time": "",
                "label": f"{venue_name} {race_num:02d}R | {r_title}",
            })

    races.sort(key=lambda x: (x["venue"], x["race_num"]))
    return races


def _query_odds_api(race_id: str, odds_type: str) -> dict:
    """オッズAPI（type: 1=単勝, 4=馬連, 5=ワイド, 6=馬単, 7=三連複）から生データを取得する"""
    for action in ("update", "init"):
        url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={odds_type}&action={action}"
        try:
            time.sleep(0.6)
            resp = requests.get(url, headers=HEADERS, timeout=10)
            payload = resp.json()
            data_block = payload.get("data")
            if isinstance(data_block, dict):
                odds_dict = data_block.get("odds", {}).get(str(odds_type), {})
                if isinstance(odds_dict, dict) and len(odds_dict) > 0:
                    return odds_dict
        except Exception:
            continue
    return {}


def fetch_result_win_odds(race_id: str) -> dict[int, float]:
    """確定済みレースの結果ページから確定単勝オッズを取得する"""
    # 1. race.netkeiba.com 結果ページ
    try:
        time.sleep(1.0)
        res_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = requests.get(res_url, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(decode_html(resp), "lxml")
        odds_map = {}
        for tr in soup.select("table#All_Result_Table tr.HorseList"):
            umaban_td = tr.select_one("td.Num.Txt_C") or tr.select_one("td:nth-of-type(3)")
            odds_td = tr.select_one("td.Odds.Txt_R span") or tr.select_one("td.Odds span")
            if umaban_td and odds_td:
                u_txt = umaban_td.get_text(strip=True)
                o_txt = odds_td.get_text(strip=True).replace(",", "")
                if u_txt.isdigit():
                    try:
                        val = float(o_txt)
                        if val >= 1.0:
                            odds_map[int(u_txt)] = val
                    except ValueError:
                        pass
        if odds_map:
            return odds_map
    except Exception:
        pass

    # 2. db.netkeiba.com 結果ページ
    try:
        time.sleep(1.0)
        db_url = f"https://db.netkeiba.com/race/{race_id}/"
        resp = requests.get(db_url, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(decode_html(resp), "lxml")
        table = soup.find("table", class_="race_table_01")
        if table:
            rows = table.find_all("tr")
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            idx_u = headers.index("馬番") if "馬番" in headers else 2
            idx_o = headers.index("単勝") if "単勝" in headers else 12
            odds_map = {}
            for tr in rows[1:]:
                tds = tr.find_all("td")
                if len(tds) > max(idx_u, idx_o):
                    u_txt = tds[idx_u].get_text(strip=True)
                    o_txt = tds[idx_o].get_text(strip=True).replace(",", "")
                    if u_txt.isdigit():
                        try:
                            val = float(o_txt)
                            if val >= 1.0:
                                odds_map[int(u_txt)] = val
                        except ValueError:
                            pass
            if odds_map:
                return odds_map
    except Exception:
        pass

    return {}


def fetch_all_market_odds(race_id: str) -> dict:
    """単勝・馬連・ワイド・馬単・三連複のオッズを一括自動取得する"""
    win_odds_map: dict[int, float] = {}
    pair_rows: dict[tuple[int, int], dict] = {}
    exacta_rows: list[dict] = []
    trio_rows: list[dict] = []

    # 1. 単勝（type=1）
    raw_win = _query_odds_api(race_id, "1")
    for k, v in raw_win.items():
        if k.isdigit() and isinstance(v, list) and len(v) > 0:
            try:
                val = float(str(v[0]).replace(",", ""))
                if val >= 1.0:
                    win_odds_map[int(k)] = val
            except ValueError:
                pass

    # APIで単勝が取れない場合は結果ページ（確定オッズ）から取得
    if not win_odds_map:
        win_odds_map = fetch_result_win_odds(race_id)

    # 2. 馬連（type=4）
    raw_umaren = _query_odds_api(race_id, "4")
    for k, v in raw_umaren.items():
        if len(k) == 4 and k.isdigit() and isinstance(v, list) and len(v) > 0:
            try:
                val = float(str(v[0]).replace(",", ""))
                if val >= 1.0:
                    h1, h2 = sorted((int(k[:2]), int(k[2:])))
                    pair_rows.setdefault((h1, h2), {"horse_num_1": h1, "horse_num_2": h2})["umaren_odds"] = val
            except ValueError:
                pass

    # 3. ワイド（type=5）
    raw_wide = _query_odds_api(race_id, "5")
    for k, v in raw_wide.items():
        if len(k) == 4 and k.isdigit() and isinstance(v, list) and len(v) > 0:
            try:
                val = float(str(v[0]).replace(",", ""))
                if val >= 1.0:
                    h1, h2 = sorted((int(k[:2]), int(k[2:])))
                    pair_rows.setdefault((h1, h2), {"horse_num_1": h1, "horse_num_2": h2})["wide_odds"] = val
            except ValueError:
                pass

    # 4. 馬単（type=6）
    raw_exacta = _query_odds_api(race_id, "6")
    for k, v in raw_exacta.items():
        if len(k) == 4 and k.isdigit() and isinstance(v, list) and len(v) > 0:
            try:
                val = float(str(v[0]).replace(",", ""))
                if val >= 1.0:
                    exacta_rows.append({
                        "horse_num_1": int(k[:2]),
                        "horse_num_2": int(k[2:]),
                        "exacta_odds": val,
                    })
            except ValueError:
                pass

    # 5. 三連複（type=7）
    raw_trio = _query_odds_api(race_id, "7")
    for k, v in raw_trio.items():
        if len(k) == 6 and k.isdigit() and isinstance(v, list) and len(v) > 0:
            try:
                val = float(str(v[0]).replace(",", ""))
                if val >= 1.0:
                    h1, h2, h3 = sorted((int(k[:2]), int(k[2:4]), int(k[4:])))
                    trio_rows.append({
                        "horse_num_1": h1,
                        "horse_num_2": h2,
                        "horse_num_3": h3,
                        "trio_odds": val,
                    })
            except ValueError:
                pass

    pair_df = pd.DataFrame(list(pair_rows.values())) if pair_rows else None
    exacta_df = pd.DataFrame(exacta_rows) if exacta_rows else None
    trio_df = pd.DataFrame(trio_rows) if trio_rows else None

    return {
        "win_odds_map": win_odds_map,
        "pair_odds_df": pair_df,
        "exacta_odds_df": exacta_df,
        "trio_odds_df": trio_df,
    }


def parse_weight_cell(text: str) -> tuple[int, int]:
    """'484(+4)' 形式の馬体重を (体重, 増減) に分割する"""
    s = str(text).strip()
    m = re.match(r"^(\d{3})\s*\(([+-]?\d+)\)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m_only = re.match(r"^(\d{3})$", s)
    if m_only:
        return int(m_only.group(1)), 0
    return 476, 0


def fetch_shutuba_and_odds(race_id: str, date_str: str) -> dict:
    """指定レースIDの出馬表と全券種オッズを一括取得して返す"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

    time.sleep(1.5)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    html_text = decode_html(resp)

    soup = BeautifulSoup(html_text, "lxml")

    venue = JRA_VENUE_CODE.get(race_id[4:6], "東京")
    title_el = soup.select_one("div.RaceName")
    race_title = re.sub(r"\s+", " ", title_el.get_text(strip=True)) if title_el else f"{venue} {int(race_id[-2:])}R"

    data01_el = soup.select_one("div.RaceData01")
    text01 = data01_el.get_text(" ", strip=True) if data01_el else ""

    time_m = re.search(r"(\d{1,2}:\d{2})発走", text01)
    start_time = time_m.group(1) if time_m else ""

    surface = "ダート" if "ダ" in text01 else "芝"
    dist_m = re.search(r"(\d{4})m", text01)
    distance = int(dist_m.group(1)) if dist_m else 2000

    cond_m = re.search(r"馬場\s*[:：]\s*(良|稍重|重|不良)", text01)
    condition = cond_m.group(1) if cond_m else "良"

    rows = []
    for tr in soup.select("table.Shutuba_Table tr.HorseList"):
        if "Cancel" in tr.get("class", []):
            continue

        tds = tr.find_all("td", recursive=False)
        if len(tds) < 7:
            continue

        waku_txt = tds[0].get_text(strip=True)
        umaban_txt = tds[1].get_text(strip=True)
        if not umaban_txt.isdigit():
            continue
        horse_num = int(umaban_txt)
        frame_num = int(waku_txt) if waku_txt.isdigit() else min(8, (horse_num - 1) // 2 + 1)

        horse_a = tr.select_one("span.HorseName a")
        horse_name = horse_a.get_text(strip=True) if horse_a else f"出走馬{horse_num:02d}"
        horse_id = f"H_{horse_num:02d}"
        if horse_a and horse_a.has_attr("href"):
            hm = re.search(r"/horse/(\d+)", horse_a["href"])
            if hm:
                horse_id = hm.group(1)

        weight_carried = 56.0
        for td in tds[4:7]:
            val_txt = td.get_text(strip=True)
            if re.match(r"^\d{2}\.\d$", val_txt):
                weight_carried = float(val_txt)
                break

        jockey_a = tr.select_one("td.Jockey a")
        jockey_name = re.sub(r"\s+", "", jockey_a.get_text(strip=True)) if jockey_a else "UNKNOWN_J"

        trainer_a = tr.select_one("td.Trainer a")
        trainer_name = re.sub(r"\s+", "", trainer_a.get_text(strip=True)) if trainer_a else "UNKNOWN_T"

        weight_td = tr.select_one("td.Weight")
        horse_weight, weight_diff = parse_weight_cell(weight_td.get_text(strip=True) if weight_td else "")

        odds_el = tr.select_one("td.Popular span") or tr.select_one(f"span#odds-1_{horse_num:02d}")
        odds_txt = odds_el.get_text(strip=True) if odds_el else ""
        try:
            win_odds = float(odds_txt)
        except ValueError:
            win_odds = np.nan

        past_tds = tr.select("td.Past")
        past_ranks = []
        prev1_tdiff = np.nan

        for idx_p, p_td in enumerate(past_tds[:3]):
            num_el = p_td.select_one("span.Num")
            r_str = num_el.get_text(strip=True) if num_el else ""
            r_int = int(np.clip(int(r_str), 1, 18)) if r_str.isdigit() else 9
            past_ranks.append(r_int)

            if idx_p == 0:
                p_text = p_td.get_text(" ", strip=True)
                diff_m = re.search(r"\(([+-]?\d+\.\d)\)", p_text)
                if diff_m:
                    prev1_tdiff = round(max(0.0, float(diff_m.group(1))), 2)

        while len(past_ranks) < 3:
            past_ranks.append(9)

        if np.isnan(prev1_tdiff):
            prev1_tdiff = round(max(0.0, (past_ranks[0] - 1) * 0.15), 2)

        rows.append({
            "race_id": race_id,
            "race_title": race_title,
            "start_time": start_time,
            "date": date_str,
            "venue": venue,
            "surface": surface,
            "distance": distance,
            "track_condition": condition,
            "horse_id": horse_id,
            "horse_name": horse_name,
            "frame_num": frame_num,
            "horse_num": horse_num,
            "weight_carried": weight_carried,
            "horse_weight": horse_weight,
            "weight_diff": weight_diff,
            "jockey_id": jockey_name,
            "trainer_id": trainer_name,
            "prev1_rank": past_ranks[0],
            "prev2_rank": past_ranks[1],
            "prev3_rank": past_ranks[2],
            "prev1_time_diff": prev1_tdiff,
            "win_odds": win_odds,
        })

    if not rows:
        raise ValueError("出馬表データを取得できませんでした。対象レースまたは日付を確認してください。")

    df = pd.DataFrame(rows).sort_values("horse_num").reset_index(drop=True)

    # 単勝・馬連・ワイド・馬単・三連複の実オッズを取得して反映
    market_odds = fetch_all_market_odds(race_id)
    win_map = market_odds["win_odds_map"]

    if win_map:
        df["win_odds"] = df.apply(
            lambda r: win_map.get(int(r["horse_num"]), r["win_odds"]),
            axis=1,
        )

    # 発売前などで単勝オッズが全く存在しない場合のみ、過去走実績から妥当な仮オッズを算出
    if df["win_odds"].isna().all():
        avg_r = (df["prev1_rank"] * 0.5 + df["prev2_rank"] * 0.3 + df["prev3_rank"] * 0.2).to_numpy()
        pseudo_strength = np.exp(-avg_r * 0.28)
        pseudo_probs = pseudo_strength / pseudo_strength.sum()
        df["win_odds"] = np.round(np.clip(0.80 / pseudo_probs, 1.8, 150.0), 1)
    else:
        df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce").fillna(60.0).clip(1.1, 999.9)

    return {
        "shutuba_df": df,
        "pair_odds_df": market_odds["pair_odds_df"],
        "exacta_odds_df": market_odds["exacta_odds_df"],
        "trio_odds_df": market_odds["trio_odds_df"],
        "has_real_win": len(win_map) > 0,
        "has_real_multi": (
            market_odds["pair_odds_df"] is not None
            or market_odds["exacta_odds_df"] is not None
            or market_odds["trio_odds_df"] is not None
        ),
    }