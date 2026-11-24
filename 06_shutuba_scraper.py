import base64
import json
import re
import time
import zlib
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
    raw = resp.content
    head = raw[:3500].lower()
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


def _clean_race_title(raw_title: str) -> str:
    s = re.sub(r"\s+", " ", str(raw_title or "")).strip()
    s = re.sub(r"(出馬表|レース結果|結果・払戻|レース情報).*$", "", s).strip()
    return s


def _extract_grade_suffix(soup: BeautifulSoup) -> str:
    """重賞・リステッドのグレードアイコンクラスから (G1)/(G2)/(G3)/(L) を完全一致で判定する"""
    icon = soup.select_one("div.RaceName span[class*='Icon_GradeType'], h1.RaceName span[class*='Icon_GradeType']")
    if not icon:
        return ""
    cls_list = icon.get("class", [])
    if "Icon_GradeType1" in cls_list:
        return "(G1)"
    if "Icon_GradeType2" in cls_list:
        return "(G2)"
    if "Icon_GradeType3" in cls_list:
        return "(G3)"
    if "Icon_GradeType15" in cls_list:
        return "(L)"
    if "Icon_GradeType10" in cls_list:
        return "(J.G1)"
    if "Icon_GradeType11" in cls_list:
        return "(J.G2)"
    if "Icon_GradeType12" in cls_list:
        return "(J.G3)"
    return ""


def _extract_race_title(soup: BeautifulSoup, venue: str, race_num: int) -> str:
    """出馬表・結果ページからレース名を確実に抽出する（多段フォールバック）"""
    fallback_pattern = re.compile(rf"^{re.escape(venue)}\s*{race_num}R$")

    # 1. RaceName 要素から取得
    for sel in ("div.RaceName", "h1.RaceName", "div.data_intro h1", "dl.racedata h1"):
        el = soup.select_one(sel)
        if el:
            txt = _clean_race_title(el.get_text(" ", strip=True))
            if txt and not fallback_pattern.match(txt):
                grade = _extract_grade_suffix(soup)
                if grade and grade not in txt and not re.search(r"\((G1|G2|G3|L|J\.G\d)\)", txt):
                    txt = f"{txt} {grade}"
                return txt

    # 2. <title> タグから取得
    title_tag = soup.find("title")
    if title_tag:
        t_raw = title_tag.get_text(" ", strip=True)
        head_part = re.split(r"出馬表|結果|レース情報|\|", t_raw)[0].strip()
        head_part = _clean_race_title(head_part)
        if head_part and not fallback_pattern.match(head_part) and "netkeiba" not in head_part.lower():
            return head_part

    # 3. RaceData02 のクラス表記（新馬・未勝利・1勝クラス等）から補完
    data02 = soup.select_one("div.RaceData02")
    if data02:
        spans = [sp.get_text(strip=True) for sp in data02.find_all("span") if sp.get_text(strip=True)]
        cls_keywords = ("新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "500万下", "1000万下", "1600万下")
        picked = [sp for sp in spans if any(k in sp for k in cls_keywords)]
        if picked:
            return " ".join(picked[:2])

    return f"{venue} {race_num}R"


def fetch_race_list_by_date(date_str: str) -> list[dict]:
    """指定日（YYYY-MM-DD）のJRA開催レース一覧（race_id, venue, race_num, race_title, label）を取得する"""
    ymd = date_str.replace("-", "")
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={ymd}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(decode_html(resp), "lxml")

    races = []
    seen = set()

    for li in soup.select("li.RaceList_DataItem"):
        a_tag = li.find("a", href=re.compile(r"race_id=\d{12}"))
        if not a_tag:
            continue
        m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
        if not m:
            continue
        rid = m.group(1)
        if rid in seen:
            continue
        venue_code = rid[4:6]
        if venue_code not in JRA_VENUE_CODE:
            continue
        seen.add(rid)

        venue = JRA_VENUE_CODE[venue_code]
        race_num = int(rid[10:12])
        title_el = li.select_one("span.ItemTitle")
        r_title = _clean_race_title(title_el.get_text(" ", strip=True)) if title_el else ""
        label = f"{venue} {race_num:02d}R {r_title}".strip()

        races.append({
            "race_id": rid,
            "venue": venue,
            "race_num": race_num,
            "race_title": r_title,
            "label": label,
        })

    if not races:
        db_url = f"https://db.netkeiba.com/race/list/{ymd}/"
        resp_db = requests.get(db_url, headers=HEADERS, timeout=15)
        if resp_db.status_code == 200:
            soup_db = BeautifulSoup(decode_html(resp_db), "lxml")
            for a_tag in soup_db.select('a[href*="/race/"]'):
                m = re.search(r"/race/(\d{12})/", a_tag.get("href", ""))
                if not m:
                    continue
                rid = m.group(1)
                if rid in seen:
                    continue
                venue_code = rid[4:6]
                if venue_code not in JRA_VENUE_CODE:
                    continue
                seen.add(rid)
                venue = JRA_VENUE_CODE[venue_code]
                race_num = int(rid[10:12])
                r_title = _clean_race_title(a_tag.get_text(" ", strip=True))
                label = f"{venue} {race_num:02d}R {r_title}".strip()
                races.append({
                    "race_id": rid,
                    "venue": venue,
                    "race_num": race_num,
                    "race_title": r_title,
                    "label": label,
                })

    races.sort(key=lambda x: (x["race_id"][4:6], x["race_num"]))
    return races


def _fetch_jra_odds_api(race_id: str, odds_type: int) -> dict:
    """netkeiba JRAオッズAPIから指定券種のオッズ辞書を取得する"""
    url = (
        f"https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?pid=api_get_jra_odds&input=UTF-8&output=json&race_id={race_id}&type={odds_type}&action=init&sort=odds&compress=1"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}
        payload = resp.json()
        raw_data = payload.get("data")
        if not raw_data:
            return {}
        if isinstance(raw_data, str):
            decoded = zlib.decompress(base64.b64decode(raw_data)).decode("utf-8", errors="replace")
            data_obj = json.loads(decoded)
        elif isinstance(raw_data, dict):
            data_obj = raw_data
        else:
            return {}
        odds_dict = data_obj.get("odds", {})
        return odds_dict.get(str(odds_type), {}) if isinstance(odds_dict, dict) else {}
    except Exception:
        return {}


def _parse_odds_float(val) -> float | None:
    if isinstance(val, list) and val:
        val = val[0]
    try:
        f = float(str(val).replace(",", "").strip())
        return f if f >= 1.0 else None
    except (ValueError, TypeError):
        return None


def _fetch_past_runs(race_id: str) -> dict[int, dict]:
    """5走前までの過去成績ページ（shutuba_past.html）から各馬番の過去3走着順・前走タイム差を取得する"""
    past_map: dict[int, dict] = {}
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return past_map
        soup = BeautifulSoup(decode_html(resp), "lxml")
        for tr in soup.select("tr.HorseList"):
            num_el = tr.select_one("td.Waku + td") or tr.select_one("td.Umaban")
            if not num_el:
                continue
            num_txt = num_el.get_text(strip=True)
            if not num_txt.isdigit():
                continue
            h_num = int(num_txt)

            ranks = []
            first_tdiff = 0.8
            past_tds = tr.select("td.Past")
            for idx_p, ptd in enumerate(past_tds[:3]):
                r_el = ptd.select_one("span.Num")
                if r_el and r_el.get_text(strip=True).isdigit():
                    ranks.append(int(r_el.get_text(strip=True)))
                else:
                    ranks.append(7)

                if idx_p == 0:
                    data_txt = ptd.get_text(" ", strip=True)
                    m_diff = re.search(r"\(([+-]?\d+\.\d)\)", data_txt)
                    if m_diff:
                        first_tdiff = max(0.0, float(m_diff.group(1)))

            while len(ranks) < 3:
                ranks.append(7)

            past_map[h_num] = {
                "prev1_rank": int(np.clip(ranks[0], 1, 18)),
                "prev2_rank": int(np.clip(ranks[1], 1, 18)),
                "prev3_rank": int(np.clip(ranks[2], 1, 18)),
                "prev1_time_diff": round(float(np.clip(first_tdiff, 0.0, 10.0)), 2),
            }
    except Exception:
        pass
    return past_map


def _fetch_confirmed_win_odds_from_result(race_id: str) -> dict[int, float]:
    """確定済みレースの場合、結果ページから確定単勝オッズを取得する"""
    win_map: dict[int, float] = {}
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return win_map
        soup = BeautifulSoup(decode_html(resp), "lxml")
        for tr in soup.select("table#All_Result_Table tr.HorseList"):
            num_el = tr.select_one("td.Num.Txt_C div") or tr.select_one("td:nth-of-type(3)")
            odds_el = tr.select_one("td.Odds.Txt_R span") or tr.select_one("td.Odds")
            if num_el and odds_el:
                n_txt = num_el.get_text(strip=True)
                o_val = _parse_odds_float(odds_el.get_text(strip=True))
                if n_txt.isdigit() and o_val is not None:
                    win_map[int(n_txt)] = o_val
    except Exception:
        pass
    return win_map


def fetch_shutuba_and_odds(race_id: str, date_str: str) -> dict:
    """指定race_idの出馬表データと全券種オッズ（単勝・馬連・ワイド・馬単・三連複）を一括取得する"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(decode_html(resp), "lxml")

    venue = JRA_VENUE_CODE.get(race_id[4:6], "東京")
    race_num = int(race_id[10:12])
    race_title = _extract_race_title(soup, venue, race_num)

    # レース条件（発走時刻・芝ダ・距離・馬場状態）
    rd01 = soup.select_one("div.RaceData01")
    rd01_txt = rd01.get_text(" ", strip=True) if rd01 else ""

    time_m = re.search(r"(\d{1,2}:\d{2})", rd01_txt)
    start_time = time_m.group(1) if time_m else ""

    surface = "ダート" if "ダ" in rd01_txt else "芝"
    dist_m = re.search(r"(\d{4})m", rd01_txt)
    distance = int(dist_m.group(1)) if dist_m else 2000

    cond_m = re.search(r"馬場\s*[:：]\s*(良|稍重|稍|重|不良)", rd01_txt)
    if cond_m:
        raw_c = cond_m.group(1)
        track_condition = "稍重" if raw_c == "稍" else raw_c
    else:
        track_condition = "良"

    # 過去走データ取得
    past_map = _fetch_past_runs(race_id)

    # 全券種オッズAPI取得
    win_api = _fetch_jra_odds_api(race_id, 1)
    umaren_api = _fetch_jra_odds_api(race_id, 4)
    wide_api = _fetch_jra_odds_api(race_id, 5)
    exacta_api = _fetch_jra_odds_api(race_id, 6)
    trio_api = _fetch_jra_odds_api(race_id, 7)

    win_odds_map: dict[int, float] = {}
    for k, v in win_api.items():
        if str(k).isdigit():
            ov = _parse_odds_float(v)
            if ov is not None:
                win_odds_map[int(k)] = ov

    if not win_odds_map:
        win_odds_map = _fetch_confirmed_win_odds_from_result(race_id)

    # 出走馬テーブルのパース
    rows = []
    for tr in soup.select("tr.HorseList"):
        if "Cancel" in tr.get("class", []):
            continue

        waku_el = tr.select_one("td[class*='Waku'] span") or tr.select_one("td[class*='Waku']")
        umaban_el = tr.select_one("td[class*='Umaban']")
        if not umaban_el:
            continue
        umaban_txt = umaban_el.get_text(strip=True)
        if not umaban_txt.isdigit():
            continue
        horse_num = int(umaban_txt)

        waku_txt = waku_el.get_text(strip=True) if waku_el else ""
        frame_num = int(waku_txt) if waku_txt.isdigit() else min(8, (horse_num - 1) // 2 + 1)

        name_el = tr.select_one("span.HorseName a") or tr.select_one("span.HorseName")
        horse_name = name_el.get_text(strip=True) if name_el else f"馬番{horse_num}"

        jockey_el = tr.select_one("td.Jockey a") or tr.select_one("td.Jockey")
        jockey_id = re.sub(r"\s+", "", jockey_el.get_text(strip=True)) if jockey_el else "UNKNOWN_J"

        trainer_el = tr.select_one("td.Trainer a") or tr.select_one("td.Trainer")
        trainer_id = re.sub(r"\s+", "", trainer_el.get_text(strip=True)) if trainer_el else "UNKNOWN_T"

        # 斤量
        kinryo = 56.0
        for td in tr.select("td.Txt_C"):
            t_val = td.get_text(strip=True)
            if re.match(r"^\d{2}\.\d$", t_val):
                kinryo = float(t_val)
                break

        # 馬体重・増減
        w_el = tr.select_one("td.Weight")
        w_txt = w_el.get_text(strip=True) if w_el else ""
        wm = re.match(r"^(\d{3})\s*\(([+-]?\d+)\)", w_txt)
        if wm:
            horse_weight = int(wm.group(1))
            weight_diff = int(wm.group(2))
        else:
            horse_weight, weight_diff = 476, 0

        # 単勝オッズ
        w_odds = win_odds_map.get(horse_num)
        if w_odds is None:
            o_el = tr.select_one("td.Popular span") or tr.select_one("td.Txt_R.Popular")
            if o_el:
                w_odds = _parse_odds_float(o_el.get_text(strip=True))

        past_info = past_map.get(
            horse_num,
            {"prev1_rank": 7, "prev2_rank": 7, "prev3_rank": 7, "prev1_time_diff": 0.8},
        )

        rows.append({
            "race_id": race_id,
            "date": date_str,
            "venue": venue,
            "race_title": race_title,
            "start_time": start_time,
            "surface": surface,
            "distance": distance,
            "track_condition": track_condition,
            "frame_num": frame_num,
            "horse_num": horse_num,
            "horse_name": horse_name,
            "weight_carried": kinryo,
            "horse_weight": horse_weight,
            "weight_diff": weight_diff,
            "jockey_id": jockey_id,
            "trainer_id": trainer_id,
            "prev1_rank": past_info["prev1_rank"],
            "prev2_rank": past_info["prev2_rank"],
            "prev3_rank": past_info["prev3_rank"],
            "prev1_time_diff": past_info["prev1_time_diff"],
            "win_odds": w_odds,
        })

    if not rows:
        raise ValueError("出馬表の馬データを取得できませんでした。発売前またはレースIDを確認してください。")

    shutuba_df = pd.DataFrame(rows)
    has_real_win = bool(shutuba_df["win_odds"].notna().any())

    if not has_real_win:
        avg_r = shutuba_df[["prev1_rank", "prev2_rank", "prev3_rank"]].mean(axis=1)
        scores = np.exp(-0.22 * avg_r)
        probs = scores / scores.sum()
        shutuba_df["win_odds"] = np.clip(0.80 / np.maximum(probs, 1e-4), 1.5, 180.0).round(1)
    else:
        median_odds = float(shutuba_df["win_odds"].median())
        shutuba_df["win_odds"] = shutuba_df["win_odds"].fillna(median_odds)

    # 馬連・ワイドオッズDF構築
    pair_records = {}
    for k, v in umaren_api.items():
        if len(str(k)) == 4 and str(k).isdigit():
            h1, h2 = int(k[:2]), int(k[2:])
            key = (min(h1, h2), max(h1, h2))
            ov = _parse_odds_float(v)
            if ov is not None:
                pair_records.setdefault(key, {"horse_num_1": key[0], "horse_num_2": key[1]})["umaren_odds"] = ov

    for k, v in wide_api.items():
        if len(str(k)) == 4 and str(k).isdigit():
            h1, h2 = int(k[:2]), int(k[2:])
            key = (min(h1, h2), max(h1, h2))
            ov = _parse_odds_float(v)
            if ov is not None:
                pair_records.setdefault(key, {"horse_num_1": key[0], "horse_num_2": key[1]})["wide_odds"] = ov

    pair_odds_df = pd.DataFrame(list(pair_records.values())) if pair_records else None

    # 馬単オッズDF構築
    exacta_rows = []
    for k, v in exacta_api.items():
        if len(str(k)) == 4 and str(k).isdigit():
            ov = _parse_odds_float(v)
            if ov is not None:
                exacta_rows.append({"horse_num_1": int(k[:2]), "horse_num_2": int(k[2:]), "exacta_odds": ov})
    exacta_odds_df = pd.DataFrame(exacta_rows) if exacta_rows else None

    # 三連複オッズDF構築
    trio_rows = []
    for k, v in trio_api.items():
        if len(str(k)) == 6 and str(k).isdigit():
            ov = _parse_odds_float(v)
            if ov is not None:
                trio_rows.append({
                    "horse_num_1": int(k[:2]),
                    "horse_num_2": int(k[2:4]),
                    "horse_num_3": int(k[4:6]),
                    "trio_odds": ov,
                })
    trio_odds_df = pd.DataFrame(trio_rows) if trio_rows else None

    has_real_multi = any(df is not None and not df.empty for df in (pair_odds_df, exacta_odds_df, trio_odds_df))

    return {
        "shutuba_df": shutuba_df,
        "pair_odds_df": pair_odds_df,
        "exacta_odds_df": exacta_odds_df,
        "trio_odds_df": trio_odds_df,
        "has_real_win": has_real_win,
        "has_real_multi": has_real_multi,
    }