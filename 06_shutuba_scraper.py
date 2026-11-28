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


def _make_soup(html_txt: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html_txt, "lxml")
    except Exception:
        return BeautifulSoup(html_txt, "html.parser")


def decode_html(resp: requests.Response) -> str:
    raw = resp.content
    head = raw[:3500].lower()
    m = re.search(rb"charset\s*=\s*['\"]?([a-z0-9_-]+)", head)
    if m:
        enc = m.group(1).decode("ascii", errors="ignore")
        if "euc" in enc:
            return raw.decode("euc_jp", errors="replace")
        if "shift" in enc or "sjis" in enc or "cp932" in enc:
            return raw.decode("cp932", errors="replace")
        if "utf" in enc:
            try:
                return raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                pass
    for enc in ("utf-8", "euc_jp", "cp932"):
        try:
            return raw.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue
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

    for sel in ("div.RaceName", "h1.RaceName", "div.data_intro h1", "dl.racedata h1"):
        el = soup.select_one(sel)
        if el:
            txt = _clean_race_title(el.get_text(" ", strip=True))
            if txt and not fallback_pattern.match(txt):
                grade = _extract_grade_suffix(soup)
                if grade and grade not in txt and not re.search(r"\((G1|G2|G3|L|J\.G\d)\)", txt):
                    txt = f"{txt} {grade}"
                return txt

    title_tag = soup.find("title")
    if title_tag:
        t_raw = title_tag.get_text(" ", strip=True)
        head_part = re.split(r"出馬表|結果|レース情報|\|", t_raw)[0].strip()
        head_part = _clean_race_title(head_part)
        if head_part and not fallback_pattern.match(head_part) and "netkeiba" not in head_part.lower():
            return head_part

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
    soup = _make_soup(decode_html(resp))

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
            soup_db = _make_soup(decode_html(resp_db))
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


def _decode_odds_data_field(raw_data) -> dict:
    """APIレスポンスの data フィールド（辞書 / 非圧縮JSON文字列 / zlib・gzip・deflate圧縮Base64）を安全に復元する"""
    if isinstance(raw_data, dict):
        return raw_data
    if not isinstance(raw_data, str) or not raw_data.strip():
        return {}

    s = raw_data.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    try:
        raw_bytes = base64.b64decode(s)
    except Exception:
        return {}

    for wbits in (zlib.MAX_WBITS, 16 + zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            text = zlib.decompress(raw_bytes, wbits).decode("utf-8", errors="replace")
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def _fetch_jra_odds_api(race_id: str, odds_type: int) -> dict:
    """netkeiba JRAオッズAPIから指定券種のオッズ辞書を取得する（非圧縮優先＋圧縮・方式フォールバック）"""
    api_headers = {
        **HEADERS,
        "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    base_url = (
        f"https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?pid=api_get_jra_odds&input=UTF-8&output=json&race_id={race_id}&type={odds_type}"
    )

    candidate_urls = [
        f"{base_url}&action=init&compress=0",
        f"{base_url}&action=init&compress=1",
    ]
    if odds_type in (4, 5, 6, 7):
        candidate_urls.extend([
            f"{base_url}&action=init&housiki=c0&compress=0",
            f"{base_url}&action=init&housiki=c0&compress=1",
        ])

    for url in candidate_urls:
        try:
            resp = requests.get(url, headers=api_headers, timeout=10)
            if resp.status_code != 200:
                continue
            txt = resp.text.strip()
            if not txt:
                continue
            if not txt.startswith("{") and "(" in txt and txt.endswith(")"):
                txt = txt[txt.find("(") + 1 : -1].strip()
            payload = json.loads(txt)
            raw_data = payload.get("data")
            if not raw_data:
                continue
            data_obj = _decode_odds_data_field(raw_data)
            odds_dict = data_obj.get("odds", {})
            if isinstance(odds_dict, dict):
                target = odds_dict.get(str(odds_type))
                if isinstance(target, dict) and len(target) > 0:
                    return target
        except Exception:
            continue
    return {}


def _parse_odds_float(val) -> float | None:
    if isinstance(val, list) and val:
        val = val[0]
    try:
        s = str(val).replace(",", "").strip()
        if not s or "---" in s or s == "-":
            return None
        f = float(s)
        return f if f >= 1.0 else None
    except (ValueError, TypeError):
        return None


def _fetch_past_runs(race_id: str) -> tuple[dict[int, dict], dict[str, dict]]:
    """
    5走前までの過去成績ページ（shutuba_past.html）から各馬の過去3走着順・前走タイム差を取得する。
    馬番マップと正規化馬名マップの両方を返し、HTMLクラス変更に左右されず確実に紐付ける。
    """
    past_by_num: dict[int, dict] = {}
    past_by_name: dict[str, dict] = {}
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return past_by_num, past_by_name
        soup = _make_soup(decode_html(resp))
        for tr in soup.select("tr.HorseList"):
            tds = tr.find_all("td")
            num_el = (
                tr.select_one("td[class*='Umaban']")
                or tr.select_one("td[class*='Num']")
                or (tds[1] if len(tds) > 1 else None)
            )
            h_num = None
            if num_el:
                ntxt = re.sub(r"\D", "", num_el.get_text(strip=True))
                if ntxt.isdigit():
                    h_num = int(ntxt)

            name_el = tr.select_one("div.Horse02 a") or tr.select_one("span.HorseName a") or tr.select_one("a[href*='/horse/']")
            h_name_norm = re.sub(r"\s+", "", name_el.get_text(strip=True)) if name_el else ""

            ranks = []
            first_tdiff = 0.8
            past_tds = tr.select("td[class*='Past']")
            for idx_p, ptd in enumerate(past_tds[:3]):
                r_el = ptd.select_one("span.Num")
                rk_val = None
                if r_el:
                    rt = re.sub(r"\D", "", r_el.get_text(strip=True))
                    if rt.isdigit():
                        rk_val = int(rt)
                if rk_val is None:
                    d01 = ptd.select_one("div.Data01")
                    if d01:
                        d01_s = re.sub(r"\d{4}\.\d{2}\.\d{2}", "", d01.get_text(" ", strip=True))
                        m_r = re.search(r"\b(\d{1,2})\s*$", d01_s.strip())
                        if m_r:
                            rk_val = int(m_r.group(1))
                ranks.append(rk_val if rk_val is not None else 7)

                if idx_p == 0:
                    data_txt = ptd.get_text(" ", strip=True)
                    m_diff = re.search(r"\(([+-]?\d+\.\d)\)", data_txt)
                    if m_diff:
                        first_tdiff = max(0.0, float(m_diff.group(1)))

            while len(ranks) < 3:
                ranks.append(7)

            info_obj = {
                "prev1_rank": int(np.clip(ranks[0], 1, 18)),
                "prev2_rank": int(np.clip(ranks[1], 1, 18)),
                "prev3_rank": int(np.clip(ranks[2], 1, 18)),
                "prev1_time_diff": round(float(np.clip(first_tdiff, 0.0, 10.0)), 2),
            }
            if h_num is not None:
                past_by_num[h_num] = info_obj
            if h_name_norm:
                past_by_name[h_name_norm] = info_obj
    except Exception:
        pass
    return past_by_num, past_by_name


def _fetch_confirmed_win_odds_from_result(race_id: str) -> dict[int, float]:
    """確定済みレースの場合、結果ページ（race.netkeiba / db.netkeiba）から確定単勝オッズを取得する"""
    win_map: dict[int, float] = {}
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            soup = _make_soup(decode_html(resp))
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

    if win_map:
        return win_map

    try:
        db_url = f"https://db.netkeiba.com/race/{race_id}/"
        resp_db = requests.get(db_url, headers=HEADERS, timeout=12)
        if resp_db.status_code == 200:
            soup_db = _make_soup(decode_html(resp_db))
            tbl = soup_db.find("table", class_="race_table_01")
            if tbl:
                for tr in tbl.find_all("tr")[1:]:
                    tds = tr.find_all("td")
                    if len(tds) > 12:
                        n_txt = tds[2].get_text(strip=True)
                        o_val = _parse_odds_float(tds[12].get_text(strip=True))
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
    soup = _make_soup(decode_html(resp))

    venue = JRA_VENUE_CODE.get(race_id[4:6], "東京")
    race_num = int(race_id[10:12])
    race_title = _extract_race_title(soup, venue, race_num)

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

    past_by_num, past_by_name = _fetch_past_runs(race_id)

    # 全券種オッズAPI取得
    win_api = _fetch_jra_odds_api(race_id, 1)
    umaren_api = _fetch_jra_odds_api(race_id, 4)
    wide_api = _fetch_jra_odds_api(race_id, 5)
    exacta_api = _fetch_jra_odds_api(race_id, 6)
    trio_api = _fetch_jra_odds_api(race_id, 7)

    win_odds_map: dict[int, float] = {}
    for k, v in win_api.items():
        digits = re.sub(r"\D", "", str(k))
        if digits:
            ov = _parse_odds_float(v)
            if ov is not None:
                win_odds_map[int(digits)] = ov

    if not win_odds_map:
        win_odds_map = _fetch_confirmed_win_odds_from_result(race_id)

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
        horse_name_norm = re.sub(r"\s+", "", horse_name)

        jockey_el = tr.select_one("td.Jockey a") or tr.select_one("td.Jockey")
        jockey_id = re.sub(r"\s+", "", jockey_el.get_text(strip=True)) if jockey_el else "UNKNOWN_J"

        trainer_el = tr.select_one("td.Trainer a") or tr.select_one("td.Trainer")
        trainer_id = re.sub(r"\s+", "", trainer_el.get_text(strip=True)) if trainer_el else "UNKNOWN_T"

        kinryo = 56.0
        for td in tr.select("td.Txt_C"):
            t_val = td.get_text(strip=True)
            if re.match(r"^\d{2}\.\d$", t_val):
                kinryo = float(t_val)
                break

        w_el = tr.select_one("td.Weight")
        w_txt = w_el.get_text(strip=True) if w_el else ""
        wm = re.match(r"^(\d{3})\s*\(([+-]?\d+)\)", w_txt)
        if wm:
            horse_weight = int(wm.group(1))
            weight_diff = int(wm.group(2))
        else:
            horse_weight, weight_diff = 476, 0

        w_odds = win_odds_map.get(horse_num)
        if w_odds is None:
            o_el = tr.select_one("td.Popular span") or tr.select_one("td.Txt_R.Popular")
            if o_el:
                w_odds = _parse_odds_float(o_el.get_text(strip=True))

        past_info = (
            past_by_name.get(horse_name_norm)
            or past_by_num.get(horse_num)
            or {"prev1_rank": 7, "prev2_rank": 7, "prev3_rank": 7, "prev1_time_diff": 0.8}
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
        d = re.sub(r"\D", "", str(k))
        if len(d) == 4:
            h1, h2 = int(d[:2]), int(d[2:])
            key = (min(h1, h2), max(h1, h2))
            ov = _parse_odds_float(v)
            if ov is not None:
                pair_records.setdefault(key, {"horse_num_1": key[0], "horse_num_2": key[1]})["umaren_odds"] = ov

    for k, v in wide_api.items():
        d = re.sub(r"\D", "", str(k))
        if len(d) == 4:
            h1, h2 = int(d[:2]), int(d[2:])
            key = (min(h1, h2), max(h1, h2))
            ov = _parse_odds_float(v)
            if ov is not None:
                pair_records.setdefault(key, {"horse_num_1": key[0], "horse_num_2": key[1]})["wide_odds"] = ov

    pair_odds_df = pd.DataFrame(list(pair_records.values())) if pair_records else None

    # 馬単オッズDF構築
    exacta_rows = []
    for k, v in exacta_api.items():
        d = re.sub(r"\D", "", str(k))
        if len(d) == 4:
            ov = _parse_odds_float(v)
            if ov is not None:
                exacta_rows.append({"horse_num_1": int(d[:2]), "horse_num_2": int(d[2:]), "exacta_odds": ov})
    exacta_odds_df = pd.DataFrame(exacta_rows) if exacta_rows else None

    # 三連複オッズDF構築
    trio_rows = []
    for k, v in trio_api.items():
        d = re.sub(r"\D", "", str(k))
        if len(d) == 6:
            ov = _parse_odds_float(v)
            if ov is not None:
                trio_rows.append({
                    "horse_num_1": int(d[:2]),
                    "horse_num_2": int(d[2:4]),
                    "horse_num_3": int(d[4:6]),
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