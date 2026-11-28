from datetime import datetime
import html
import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/",
}

VENUE_RE = re.compile(
    r"(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉|大井|川崎|船橋|浦和|門別|盛岡|水沢|金沢|笠松|名古屋|園田|姫路|高知|佐賀|帯広|海外)"
)


def _make_soup(html_txt: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html_txt, "lxml")
    except Exception:
        return BeautifulSoup(html_txt, "html.parser")


def _decode_html(resp: requests.Response) -> str:
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


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip()


def _norm_jockey_key(s: str) -> str:
    """騎手名の表記揺れ（L トーレス / トーレス、Cルメール / ルメール、▲減量記号など）を吸収して比較用キーを返す"""
    t = re.sub(r"\s+", "", str(s or "")).strip()
    t = re.sub(r"^[▲△☆◇★]+", "", t)
    t = re.sub(r"^[A-Za-zＡ-Ｚ．.]+", "", t)
    return t


def _short_date(d_str: str, ref_year: str = "") -> str:
    s = str(d_str or "").strip().replace(".", "/")
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if not m:
        return s or "-"
    y, mo, dy = m.group(1), int(m.group(2)), int(m.group(3))
    if ref_year and y == ref_year:
        return f"{mo:02d}/{dy:02d}"
    return f"{y[2:]}/{mo:02d}/{dy:02d}"


def _parse_date_obj(d_str: str):
    s = str(d_str or "").strip().replace(".", "/")
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _clean_venue(v_str: str) -> str:
    s = str(v_str or "").strip()
    m = VENUE_RE.search(s)
    if m:
        return m.group(1)
    s2 = re.sub(r"\d+", "", s).strip()
    return s2[:4] if s2 else "-"


def _season_label(date_str: str) -> str:
    s = str(date_str or "").strip().replace(".", "/")
    m = re.match(r"^\d{4}/(\d{1,2})/\d{1,2}$", s)
    if not m:
        return ""
    mo = int(m.group(1))
    if mo in (3, 4, 5):
        return "春(3〜5月)"
    if mo in (6, 7, 8):
        return "夏(6〜8月)"
    if mo in (9, 10, 11):
        return "秋(9〜11月)"
    return "冬(12〜2月)"


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_race_horse_meta(race_id: str) -> dict:
    """
    shutuba.html から全馬の horse_id・性齢（例: 牡4）・馬名マップを一括取得してキャッシュする。
    """
    meta_by_num: dict[int, dict] = {}
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return meta_by_num
        soup = _make_soup(_decode_html(resp))
        for tr in soup.select("tr.HorseList"):
            tds = tr.find_all("td")
            num_el = tr.select_one("td[class*='Umaban']")
            if num_el is None and len(tds) > 1:
                num_el = tds[1]
            if not num_el:
                continue
            ntxt = re.sub(r"\D", "", num_el.get_text(strip=True))
            if not ntxt.isdigit():
                continue
            hnum = int(ntxt)

            h_id = ""
            h_name = ""
            a_el = tr.select_one("span.HorseName a") or tr.select_one("a[href*='/horse/']")
            if a_el:
                h_name = a_el.get_text(strip=True)
                hm = re.search(r"/horse/([a-zA-Z0-9]+)", a_el.get("href", ""))
                if hm:
                    h_id = hm.group(1)

            barei_el = tr.select_one("td.Barei")
            sex_age = barei_el.get_text(strip=True) if barei_el else ""
            if not sex_age:
                for td in tds:
                    tt = td.get_text(strip=True)
                    if re.match(r"^[牡牝セ]\d+$", tt):
                        sex_age = tt
                        break

            meta_by_num[hnum] = {
                "horse_id": h_id,
                "horse_name": h_name,
                "sex_age": sex_age,
            }
    except Exception:
        pass
    return meta_by_num


def _parse_db_results_table(soup: BeautifulSoup, max_runs: int = 40) -> list[dict]:
    """db.netkeiba.com の table.db_h_race_results から全過去走（最大max_runs走）を列名ベースで抽出する"""
    tbl = soup.select_one("table.db_h_race_results")
    if not tbl:
        return []

    header_row = tbl.find("tr")
    if not header_row:
        return []
    ths = [re.sub(r"\s+", "", th.get_text(strip=True)) for th in header_row.find_all(["th", "td"])]
    col_map = {name: i for i, name in enumerate(ths)}

    def _cell(tds, key_candidates: tuple[str, ...]) -> str:
        for k in key_candidates:
            if k in col_map and col_map[k] < len(tds):
                return re.sub(r"\s+", " ", tds[col_map[k]].get_text(" ", strip=True)).strip()
        return ""

    runs = []
    for tr in tbl.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue
        date_raw = _cell(tds, ("日付",))
        if not re.match(r"^\d{4}/\d{1,2}/\d{1,2}$", date_raw):
            continue

        rank_raw = _cell(tds, ("着順",))
        if not rank_raw:
            continue
        head_raw = re.sub(r"\D", "", _cell(tds, ("頭数",)))
        pop_raw = re.sub(r"\D", "", _cell(tds, ("人気",)))
        venue_raw = _clean_venue(_cell(tds, ("開催",)))
        race_name = _cell(tds, ("レース名",))
        jockey = _cell(tds, ("騎手",))
        kinryo = _cell(tds, ("斤量",))
        dist_raw = _cell(tds, ("距離",))
        cond_raw = _cell(tds, ("馬場",))
        time_raw = _cell(tds, ("タイム",))
        diff_raw = _cell(tds, ("着差",))
        pass_raw = _cell(tds, ("通過",))
        agari_raw = _cell(tds, ("上り", "上がり"))
        wt_raw = _cell(tds, ("馬体重",))
        rival_raw = _cell(tds, ("勝ち馬(2着馬)", "勝ち馬"))

        runs.append({
            "date_full": date_raw,
            "venue": venue_raw,
            "race_name": race_name or "-",
            "distance": dist_raw or "-",
            "condition": cond_raw or "-",
            "rank": rank_raw,
            "head": head_raw,
            "popularity": pop_raw,
            "time": time_raw or "-",
            "diff": diff_raw or "-",
            "passing": pass_raw or "-",
            "agari": agari_raw or "-",
            "jockey": jockey or "-",
            "kinryo": kinryo or "-",
            "weight": wt_raw or "-",
            "rival": rival_raw or "-",
        })
        if len(runs) >= max_runs:
            break
    return runs


def _parse_shutuba_past_for_horse(race_id: str, horse_num: int, horse_name: str) -> dict:
    """
    race.netkeiba.com/race/shutuba_past.html から対象馬の過去5走と基本情報を抽出するフォールバック。
    """
    out = {"horse_id": "", "sire": "", "bms": "", "sex_age": "", "runs": []}
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return out
        soup = _make_soup(_decode_html(resp))
        target_tr = None
        norm_target = _norm_name(horse_name)

        rows = soup.select("tr.HorseList")
        if norm_target:
            for tr in rows:
                row_txt = _norm_name(tr.get_text(" ", strip=True))
                if norm_target in row_txt:
                    target_tr = tr
                    break

        if target_tr is None:
            for tr in rows:
                tds = tr.find_all("td")
                cand_els = [
                    tr.select_one("td[class*='Umaban']"),
                    tr.select_one("td[class*='Num']"),
                    tds[1] if len(tds) > 1 else None,
                ]
                for el in cand_els:
                    if el is None:
                        continue
                    nt = re.sub(r"\D", "", el.get_text(strip=True))
                    if nt.isdigit() and int(nt) == int(horse_num):
                        target_tr = tr
                        break
                if target_tr is not None:
                    break

        if target_tr is None:
            return out

        h_link = target_tr.select_one("a[href*='/horse/']")
        if h_link:
            hm = re.search(r"/horse/([a-zA-Z0-9]+)", h_link.get("href", ""))
            if hm:
                out["horse_id"] = hm.group(1)

        info_td = target_tr.select_one("td.Horse_Info") or target_tr
        info_txt = info_td.get_text(" ", strip=True)
        m_sa = re.search(r"([牡牝セ]\d+)", info_txt)
        if m_sa:
            out["sex_age"] = m_sa.group(1)

        h01 = target_tr.select_one("div.Horse01")
        if h01:
            out["sire"] = h01.get_text(" ", strip=True).strip()
        h03 = target_tr.select_one("div.Horse03")
        if h03:
            bms_txt = h03.get_text(" ", strip=True)
            m_bms = re.search(r"\(([^)]+)\)", bms_txt)
            if m_bms:
                out["bms"] = m_bms.group(1).strip()

        past_tds = target_tr.select("td[class*='Past']")
        runs = []
        for ptd in past_tds[:5]:
            full_txt = re.sub(r"\s+", " ", ptd.get_text(" ", strip=True))
            if not full_txt or full_txt in ("-", "--") or not re.search(r"\d{4}\.\d{2}\.\d{2}", full_txt):
                continue

            d01 = ptd.select_one("div.Data01")
            d02 = ptd.select_one("div.Data02")
            d03 = ptd.select_one("div.Data03")
            d05 = ptd.select_one("div.Data05")
            d06 = ptd.select_one("div.Data06")
            d07 = ptd.select_one("div.Data07")

            d01_txt = re.sub(r"\s+", " ", d01.get_text(" ", strip=True)) if d01 else full_txt
            m_date = re.search(r"(\d{4}\.\d{2}\.\d{2})", d01_txt)
            date_full = m_date.group(1).replace(".", "/") if m_date else ""

            rank_el = ptd.select_one("span.Num")
            rank_str = rank_el.get_text(strip=True) if rank_el else ""
            if not rank_str and d01:
                m_rk = re.search(r"\b(\d{1,2})\s*$", d01_txt)
                rank_str = m_rk.group(1) if m_rk else "-"

            venue_val = _clean_venue(d01_txt)
            race_name = re.sub(r"\s+", " ", d02.get_text(" ", strip=True)) if d02 else "-"

            d03_txt = re.sub(r"\s+", " ", d03.get_text(" ", strip=True)) if d03 else full_txt
            m_head = re.search(r"(\d{1,2})頭", d03_txt)
            head_val = m_head.group(1) if m_head else ""
            m_pop = re.search(r"(\d{1,2})人", d03_txt)
            pop_val = m_pop.group(1) if m_pop else ""

            j_clean = re.sub(r"\d+頭|\d+番|\d+人|\d{2}\.\d", "", d03_txt).strip()
            jockey_val = j_clean.split()[0] if j_clean else "-"

            d05_txt = re.sub(r"\s+", " ", d05.get_text(" ", strip=True)) if d05 else full_txt
            m_dist = re.search(r"([芝ダ障]\d{3,4})", d05_txt)
            dist_val = m_dist.group(1) if m_dist else "-"
            m_time = re.search(r"(\d:\d{2}\.\d|\d{2}\.\d)", d05_txt)
            time_val = m_time.group(1) if m_time else "-"
            m_cond = re.search(r"(良|稍重|稍|重|不良)", d05_txt)
            cond_val = m_cond.group(1) if m_cond else "-"

            d06_txt = re.sub(r"\s+", " ", d06.get_text(" ", strip=True)) if d06 else full_txt
            m_agari = re.search(r"\((\d{2}\.\d)\)", d06_txt)
            agari_val = m_agari.group(1) if m_agari else "-"
            m_wt = re.search(r"(\d{3}\([+-]?\d+\)|\d{3})", d06_txt)
            wt_val = m_wt.group(1) if m_wt else "-"
            m_pass = re.search(r"^([\d-]+)", d06_txt)
            pass_val = m_pass.group(1) if m_pass else "-"

            d07_txt = re.sub(r"\s+", " ", d07.get_text(" ", strip=True)) if d07 else ""
            m_diff = re.search(r"\(([+-]?\d+\.\d)\)", d07_txt)
            diff_val = m_diff.group(1) if m_diff else "-"

            runs.append({
                "date_full": date_full,
                "venue": venue_val,
                "race_name": race_name,
                "distance": dist_val,
                "condition": cond_val,
                "rank": rank_str or "-",
                "head": head_val,
                "popularity": pop_val,
                "time": time_val,
                "diff": diff_val,
                "passing": pass_val,
                "agari": agari_val,
                "jockey": jockey_val,
                "kinryo": "-",
                "weight": wt_val,
                "rival": "-",
            })

        out["runs"] = runs
    except Exception:
        pass
    return out


@st.cache_data(ttl=600, show_spinner=False)
def fetch_horse_profile_and_past5(race_id: str, horse_num: int, horse_name: str) -> dict:
    """
    4段フォールバックで個別馬の基本プロフィール（性齢・厩舎・血統・通算成績）と全戦績を取得する。
    """
    info = {
        "horse_id": "",
        "horse_url": "",
        "sex_age": "",
        "trainer": "",
        "sire": "",
        "bms": "",
        "career": "",
        "runs": [],
    }

    meta_map = _fetch_race_horse_meta(race_id)
    h_meta = meta_map.get(int(horse_num), {})
    if not h_meta and horse_name:
        norm_t = _norm_name(horse_name)
        for v in meta_map.values():
            if _norm_name(v.get("horse_name", "")) == norm_t:
                h_meta = v
                break

    horse_id = str(h_meta.get("horse_id", "") or "")
    info["sex_age"] = str(h_meta.get("sex_age", "") or "")

    past_fb = None
    if not horse_id:
        past_fb = _parse_shutuba_past_for_horse(race_id, horse_num, horse_name)
        horse_id = past_fb.get("horse_id", "")
        if not info["sex_age"]:
            info["sex_age"] = past_fb.get("sex_age", "")
        info["sire"] = past_fb.get("sire", "")
        info["bms"] = past_fb.get("bms", "")
        info["runs"] = past_fb.get("runs", [])

    if horse_id:
        info["horse_id"] = horse_id
        info["horse_url"] = f"https://db.netkeiba.com/horse/{horse_id}/"
        db_headers = {**HEADERS, "Referer": "https://db.netkeiba.com/"}

        try:
            resp = requests.get(info["horse_url"], headers=db_headers, timeout=12)
            if resp.status_code == 200:
                soup = _make_soup(_decode_html(resp))

                if not info["sex_age"]:
                    p01 = soup.select_one("div.horse_title p.txt_01") or soup.select_one("p.txt_01")
                    if p01:
                        m_sa = re.search(r"([牡牝セ]\d+)", p01.get_text(" ", strip=True))
                        if m_sa:
                            info["sex_age"] = m_sa.group(1)

                prof_tbl = soup.select_one("table.db_prof_table")
                if prof_tbl:
                    for tr in prof_tbl.find_all("tr"):
                        th = tr.find("th")
                        td = tr.find("td")
                        if not th or not td:
                            continue
                        k_txt = th.get_text(strip=True)
                        v_txt = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
                        if "調教師" in k_txt and v_txt:
                            info["trainer"] = v_txt
                        elif "通算成績" in k_txt and v_txt:
                            info["career"] = v_txt

                blood_tbl = soup.select_one("table.blood_table")
                if blood_tbl:
                    b_trs = blood_tbl.find_all("tr")
                    if len(b_trs) >= 1:
                        td0 = b_trs[0].find("td")
                        if td0:
                            a0 = td0.find("a")
                            info["sire"] = a0.get_text(strip=True) if a0 else td0.get_text(" ", strip=True).split()[0]
                    if len(b_trs) >= 3:
                        tds2 = b_trs[2].find_all("td")
                        if len(tds2) >= 2:
                            a_bms = tds2[1].find("a")
                            info["bms"] = a_bms.get_text(strip=True) if a_bms else tds2[1].get_text(" ", strip=True).split()[0]

                db_runs = _parse_db_results_table(soup, max_runs=40)
                if db_runs:
                    info["runs"] = db_runs
        except Exception:
            pass

        if not info["runs"]:
            ajax_headers = {
                **db_headers,
                "Referer": info["horse_url"],
                "X-Requested-With": "XMLHttpRequest",
            }
            for sub_url in (
                f"https://db.netkeiba.com/horse/ajax_horse_results.html?id={horse_id}",
                f"https://db.netkeiba.com/horse/result/{horse_id}/",
            ):
                try:
                    resp_s = requests.get(sub_url, headers=ajax_headers, timeout=12)
                    if resp_s.status_code != 200:
                        continue
                    soup_s = _make_soup(_decode_html(resp_s))
                    sub_runs = _parse_db_results_table(soup_s, max_runs=40)
                    if sub_runs:
                        info["runs"] = sub_runs
                        break
                except Exception:
                    continue

    if not info["runs"] or not info["sire"]:
        if past_fb is None:
            past_fb = _parse_shutuba_past_for_horse(race_id, horse_num, horse_name)
        if not info["runs"] and past_fb.get("runs"):
            info["runs"] = past_fb["runs"]
        if not info["sire"] and past_fb.get("sire"):
            info["sire"] = past_fb["sire"]
        if not info["bms"] and past_fb.get("bms"):
            info["bms"] = past_fb["bms"]

    return info


def summarize_horse_tendencies(
    all_runs: list[dict],
    target_venue: str,
    target_surface: str,
    target_distance: int,
    target_date: str,
    target_cond: str,
) -> dict:
    """
    取得した全戦績からコース適性・距離適性・季節傾向・馬場状態・脚質傾向を自動分析し、
    一目でわかる概要テキストと適性バッジを生成する。
    """
    valid = []
    for r in all_runs:
        rk_s = str(r.get("rank", "")).strip()
        if not rk_s.isdigit():
            continue
        valid.append({
            "date": str(r.get("date_full", "")),
            "venue": str(r.get("venue", "")),
            "distance": str(r.get("distance", "")),
            "condition": str(r.get("condition", "")),
            "rank": int(rk_s),
            "passing": str(r.get("passing", "")),
            "agari": str(r.get("agari", "")),
        })

    if not valid:
        return {
            "headline": "戦績データが少ないため、近走の着順・騎手気配・調教を中心にご確認ください。",
            "bullets": [],
            "badges": [],
        }

    def _stats(sub: list[dict]) -> tuple[int, int, int, int, float]:
        n = len(sub)
        w = sum(1 for x in sub if x["rank"] == 1)
        s = sum(1 for x in sub if x["rank"] == 2)
        t = sum(1 for x in sub if x["rank"] == 3)
        p_rate = (w + s + t) / n if n > 0 else 0.0
        return w, s, t, n - (w + s + t), p_rate

    bullets = []
    badges = []
    headline_parts = []

    # 1. 競馬場（コース）適性
    by_venue: dict[str, list[dict]] = {}
    for x in valid:
        v = x["venue"]
        if v and v != "-":
            by_venue.setdefault(v, []).append(x)

    cur_v_runs = by_venue.get(target_venue, [])
    if cur_v_runs:
        w, s, t, o, pr = _stats(cur_v_runs)
        rec = f"[{w}-{s}-{t}-{o}]"
        if pr >= 0.50 and len(cur_v_runs) >= 2:
            headline_parts.append(f"{target_venue}コース巧者{rec}")
            bullets.append(f"舞台適性：今回の{target_venue}コースは {rec}（複勝率{int(round(pr * 100))}%）と好相性です。")
            badges.append((f"{target_venue}得意 {rec}", "#047857", "#ecfdf5"))
        elif (w + s + t) == 0 and len(cur_v_runs) >= 3:
            bullets.append(f"舞台適性：今回の{target_venue}コースは {rec} とやや苦戦傾向です。")
            badges.append((f"{target_venue}苦戦 {rec}", "#b45309", "#fffbeb"))
        else:
            bullets.append(f"舞台適性：今回の{target_venue}コースは {rec}。")
    else:
        bullets.append(f"舞台適性：{target_venue}コースは今回が初出走です。")

    best_v = None
    best_v_score = (-1.0, 0)
    for v, sub in by_venue.items():
        if len(sub) < 2:
            continue
        w, s, t, o, pr = _stats(sub)
        if pr >= 0.50 and (pr, len(sub)) > best_v_score:
            best_v_score = (pr, len(sub))
            best_v = (v, w, s, t, o, pr)
    if best_v and best_v[0] != target_venue:
        v, w, s, t, o, pr = best_v
        bullets.append(f"得意コース：{v}コースで [{w}-{s}-{t}-{o}]（複勝率{int(round(pr * 100))}%）と良績を残しています。")
        if not headline_parts:
            headline_parts.append(f"{v}コース好走歴あり[{w}-{s}-{t}-{o}]")

    # 2. 距離・トラック適性
    surf_short = "ダ" if "ダ" in str(target_surface) else "芝"
    near_dist_runs = []
    for x in valid:
        m = re.search(r"([芝ダ障])(\d{3,4})", x["distance"])
        if m and m.group(1) == surf_short and abs(int(m.group(2)) - int(target_distance)) <= 200:
            near_dist_runs.append(x)
    if near_dist_runs:
        w, s, t, o, pr = _stats(near_dist_runs)
        rec = f"[{w}-{s}-{t}-{o}]"
        if pr >= 0.50 and len(near_dist_runs) >= 2:
            bullets.append(f"距離適性：{surf_short}{target_distance}m前後（±200m）で {rec} と安定して走れています。")
            badges.append((f"{surf_short}{target_distance}m前後 {rec}", "#1d4ed8", "#eff6ff"))
            if not headline_parts:
                headline_parts.append(f"{surf_short}{target_distance}m前後で安定{rec}")
        else:
            bullets.append(f"距離適性：{surf_short}{target_distance}m前後（±200m）は {rec}。")

    # 3. 季節・時期傾向
    by_season: dict[str, list[dict]] = {}
    for x in valid:
        sea = _season_label(x["date"])
        if sea:
            by_season.setdefault(sea, []).append(x)

    cur_sea = _season_label(target_date.replace("-", "/"))
    best_sea = None
    best_sea_score = (-1.0, 0)
    for sea, sub in by_season.items():
        if len(sub) < 2:
            continue
        w, s, t, o, pr = _stats(sub)
        if pr >= 0.50 and (pr, len(sub)) > best_sea_score:
            best_sea_score = (pr, len(sub))
            best_sea = (sea, w, s, t, o, pr)

    if best_sea:
        sea, w, s, t, o, pr = best_sea
        if sea == cur_sea:
            bullets.append(f"季節傾向：今走と同じ{sea}は [{w}-{s}-{t}-{o}]（複勝率{int(round(pr * 100))}%）と好走シーズンです。")
            headline_parts.append(f"{sea}に好走集中[{w}-{s}-{t}-{o}]")
            badges.append((f"{sea}好調 [{w}-{s}-{t}-{o}]", "#7c3aed", "#f5f3ff"))
        else:
            bullets.append(f"季節傾向：{sea}に [{w}-{s}-{t}-{o}]（複勝率{int(round(pr * 100))}%）と実績が集中しています。")

    # 4. 馬場状態（良 vs 道悪）
    heavy_runs = [x for x in valid if any(k in x["condition"] for k in ("稍", "重", "不"))]
    if len(heavy_runs) >= 2:
        w, s, t, o, pr = _stats(heavy_runs)
        if pr >= 0.50:
            bullets.append(f"馬場適性：道悪（稍重〜不良）でも [{w}-{s}-{t}-{o}] とパワーを要する馬場を苦にしません。")
            badges.append((f"道悪OK [{w}-{s}-{t}-{o}]", "#0369a1", "#f0f9ff"))
        elif (w + s + t) == 0 and len(heavy_runs) >= 3:
            bullets.append(f"馬場適性：道悪（稍重〜不良）は [{w}-{s}-{t}-{o}] と良馬場向きです。")

    # 5. 脚質傾向（通過順から推定）
    first_corners = []
    for x in valid[:8]:
        m_p = re.match(r"^(\d+)", x["passing"].strip())
        if m_p:
            first_corners.append(int(m_p.group(1)))
    if len(first_corners) >= 2:
        avg_pos = float(np.mean(first_corners))
        if avg_pos <= 2.5:
            style_txt = "逃げ・先行タイプ（前目で流れに乗れる脚質）"
            badges.append(("逃げ・先行", "#dc2626", "#fef2f2"))
        elif avg_pos <= 5.5:
            style_txt = "好位〜中団タイプ（自在に立ち回れる脚質）"
            badges.append(("好位・自在", "#d97706", "#fffbeb"))
        else:
            style_txt = "差し・追込タイプ（終いの末脚を活かす脚質）"
            badges.append(("差し・追込", "#2563eb", "#eff6ff"))
        bullets.append(f"脚質傾向：{style_txt}です。")

    if headline_parts:
        headline = "この馬は " + " ／ ".join(headline_parts[:2]) + " の傾向があります。"
    else:
        w, s, t, o, pr = _stats(valid)
        headline = f"集計戦績 [{w}-{s}-{t}-{o}]（複勝率{int(round(pr * 100))}%）。大きな偏りは少なく条件を問わず走るタイプです。"

    return {
        "headline": headline,
        "bullets": bullets[:5],
        "badges": badges[:4],
    }


def analyze_feature_contributions(
    race_df: pd.DataFrame,
    target_horse_num: int,
    model,
    feature_cols: list[str],
    category_maps: dict,
    train_model,
) -> list[dict]:
    """
    対象馬の特徴量グループを「同レース出走馬の平均／最頻値」に1グループずつ置換し、
    AI生勝率の変化量（プラス寄与・マイナス寄与）を実モデルから直接算出する。
    """
    df_reset = race_df.reset_index(drop=True)
    feat_df, _ = train_model.create_features(df_reset, category_maps=category_maps)
    feat_df = feat_df.reset_index(drop=True)
    base_raw = np.clip(model.predict_proba(feat_df[feature_cols])[:, 1], 1e-6, 1.0)
    base_probs = base_raw / base_raw.sum()

    matches = np.where(df_reset["horse_num"].astype(int).to_numpy() == int(target_horse_num))[0]
    if len(matches) == 0:
        return []
    pos = int(matches[0])
    target_base_prob = float(base_probs[pos])

    row_src = df_reset.iloc[pos]
    p1_v = int(row_src.get("prev1_rank", 7))
    p2_v = int(row_src.get("prev2_rank", 7))
    p3_v = int(row_src.get("prev3_rank", 7))
    td_v = float(row_src.get("prev1_time_diff", 0.8))
    jk_v = str(row_src.get("jockey_id", "-"))
    tr_v = str(row_src.get("trainer_id", "-"))
    fr_v = int(row_src.get("frame_num", 1))
    hn_v = int(row_src.get("horse_num", 1))
    wc_v = float(row_src.get("weight_carried", 56.0))
    hw_v = int(row_src.get("horse_weight", 476))
    wd_v = int(row_src.get("weight_diff", 0))

    jockey_cols = [
        c for c in (
            "jockey_id", "jockey_win_rate", "jockey_place_rate",
            "jockey_surface_win_rate", "jockey_venue_win_rate",
            "rel_jockey_win_rate", "jockey_upgrade",
        )
        if c in feature_cols
    ]
    trainer_cols = [
        c for c in ("trainer_id", "trainer_win_rate", "trainer_place_rate")
        if c in feature_cols
    ]

    groups = [
        {
            "name": "近3走の着順",
            "cols": ["prev1_rank", "prev2_rank", "prev3_rank", "avg_prev_rank", "best_prev_rank", "rank_trend", "rel_prev_rank"],
            "desc": f"{p1_v}着→{p2_v}着→{p3_v}着",
        },
        {
            "name": "前走タイム差",
            "cols": ["prev1_time_diff", "rel_time_diff"],
            "desc": f"{td_v:.2f}秒差",
        },
        {
            "name": "騎手実績・適性",
            "cols": jockey_cols,
            "desc": jk_v,
        },
        {
            "name": "厩舎力",
            "cols": trainer_cols,
            "desc": tr_v,
        },
        {
            "name": "枠順・馬番",
            "cols": ["frame_num", "horse_num", "frame_ratio"],
            "desc": f"{fr_v}枠{hn_v}番",
        },
        {
            "name": "斤量・馬体",
            "cols": ["weight_carried", "horse_weight", "weight_diff", "abs_weight_diff", "weight_ratio"],
            "desc": f"{wc_v:.1f}kg / {hw_v}kg({wd_v:+d})",
        },
    ]

    contribs = []
    for g in groups:
        mod_feat = feat_df[feature_cols].copy()
        valid_cols = [c for c in g["cols"] if c in mod_feat.columns]
        if not valid_cols:
            continue

        for c in valid_cols:
            col_idx = mod_feat.columns.get_loc(c)
            if isinstance(mod_feat[c].dtype, pd.CategoricalDtype):
                mode_vals = mod_feat[c].mode()
                if len(mode_vals) > 0:
                    mod_feat.iloc[pos, col_idx] = mode_vals.iloc[0]
            else:
                med_val = float(pd.to_numeric(mod_feat[c], errors="coerce").median())
                if np.isnan(med_val):
                    continue
                if pd.api.types.is_integer_dtype(mod_feat[c].dtype):
                    mod_feat.iloc[pos, col_idx] = int(round(med_val))
                else:
                    mod_feat[c] = mod_feat[c].astype(float)
                    mod_feat.iloc[pos, col_idx] = float(med_val)

        alt_raw = np.clip(model.predict_proba(mod_feat)[:, 1], 1e-6, 1.0)
        alt_probs = alt_raw / alt_raw.sum()
        delta_pct = (target_base_prob - float(alt_probs[pos])) * 100.0

        contribs.append({
            "factor": g["name"],
            "detail": g["desc"],
            "delta_pct": delta_pct,
        })

    contribs.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)
    return contribs


def build_signal_explanation(
    row: pd.Series,
    race_df: pd.DataFrame,
    ev_threshold: float,
    exclude_fav1: bool,
) -> dict:
    """シグナル（軸・穴注目・紐候補・様子見）と信頼度の判定理由を生成する"""
    ai_rank = int(row["ai_rank"])
    pop = int(row["popularity"])
    win_p = float(row["pred_win_pct"])
    quin_p = float(row["pred_quinella_pct"])
    place_p = float(row["pred_place_pct"])
    ev = float(row["単勝期待値(EV)"])
    odds = float(row["win_odds"])
    ability = float(row["ability_score"])

    mkt_implied_pct = (0.80 / max(odds, 1.05)) * 100.0
    gap_pct = win_p - mkt_implied_pct

    if ai_rank == 1 and not (exclude_fav1 and pop == 1):
        sig_label = "軸"
        sig_color = "#dc2626"
        sig_bg = "#fef2f2"
        main_reason = (
            f"出走{len(race_df)}頭中でAI勝率1位（{win_p:.1f}%・能力値{ability:.1f}）、"
            f"3着内率も{place_p:.1f}%と抜けており、馬券の中心『軸』判定。"
        )
    elif ev >= ev_threshold and win_p >= 4.0:
        sig_label = "穴注目"
        sig_color = "#c2410c"
        sig_bg = "#fff7ed"
        main_reason = (
            f"{pop}番人気（{odds:.1f}倍・市場評価{mkt_implied_pct:.1f}%）に対しAI勝率{win_p:.1f}%。"
            f"単勝EV {ev:.2f}（基準{ev_threshold:.2f}超）で妙味抜群のため『穴注目』判定。"
        )
    elif ai_rank <= 6 or place_p >= 25.0:
        sig_label = "紐候補"
        sig_color = "#2563eb"
        sig_bg = "#eff6ff"
        if ai_rank == 1 and exclude_fav1 and pop == 1:
            main_reason = "AI勝率1位ですが「1番人気除外」ONのため軸を見送り『紐候補』判定。"
        else:
            main_reason = (
                f"AI順位{ai_rank}位・3着内率{place_p:.1f}%と連対・複勝圏内の実力があり、"
                f"連系馬券の相手として押さえる『紐候補』判定（単勝EV {ev:.2f}）。"
            )
    else:
        sig_label = "様子見"
        sig_color = "#64748b"
        sig_bg = "#f8fafc"
        main_reason = (
            f"AI順位{ai_rank}位（勝率{win_p:.1f}%・3着内率{place_p:.1f}%）、単勝EV {ev:.2f}のため"
            "現オッズでは投資妙味が薄く『様子見』判定。"
        )

    if ai_rank == 1 and win_p >= 22.0:
        conf_label = "信頼度 S"
        conf_color = "#dc2626"
    elif ai_rank == 1 or (ai_rank == 2 and win_p >= 14.0):
        conf_label = "信頼度 A"
        conf_color = "#dc2626"
    elif ai_rank <= 3 and win_p >= 9.0:
        conf_label = "信頼度 B"
        conf_color = "#d97706"
    else:
        conf_label = ""
        conf_color = "#64748b"

    return {
        "sig_label": sig_label,
        "sig_color": sig_color,
        "sig_bg": sig_bg,
        "main_reason": main_reason,
        "conf_label": conf_label,
        "conf_color": conf_color,
        "mkt_implied_pct": mkt_implied_pct,
        "gap_pct": gap_pct,
    }


def render_horse_side_panel(
    race_df: pd.DataFrame,
    selected_horse_num: int,
    race_id: str,
    date_str: str,
    ev_threshold: float,
    exclude_fav1: bool,
    model,
    feature_cols: list[str],
    category_maps: dict,
    train_model,
):
    """
    出走表の右側に並ぶインライン馬詳細パネルを描画する。
    「‹ 前の馬 | 1 / N | 次の馬 ›」「✕」「シグナル根拠」「特徴・適性サマリー」「基本」「近5走」を表示。
    """
    ordered_df = race_df.sort_values("ai_rank", ascending=True).reset_index(drop=True)
    total_n = len(ordered_df)
    matches = ordered_df.index[ordered_df["horse_num"].astype(int) == int(selected_horse_num)].tolist()
    cur_idx = matches[0] if matches else 0
    row = ordered_df.iloc[cur_idx]

    h_num = int(row["horse_num"])
    h_name = str(row["horse_name"])
    jockey_now = str(row.get("jockey_id", "-"))
    kinryo_now = float(row.get("weight_carried", 56.0))
    trainer_now = str(row.get("trainer_id", "-"))
    venue_now = str(row.get("venue", "東京"))
    surf_now = str(row.get("surface", "芝"))
    dist_now = int(float(row.get("distance", 2000)))
    cond_now = str(row.get("track_condition", "良"))
    pop_now = int(row["popularity"])
    odds_now = float(row["win_odds"])
    win_p = float(row["pred_win_pct"])
    quin_p = float(row["pred_quinella_pct"])
    place_p = float(row["pred_place_pct"])
    ev_now = float(row["単勝期待値(EV)"])

    exp = build_signal_explanation(row, ordered_df, ev_threshold=ev_threshold, exclude_fav1=exclude_fav1)

    with st.container(border=True):
        # 1. ヘッダー行（馬番・馬名・モデルバッジ・✕閉じる）
        h_c1, h_c2 = st.columns([5.2, 1.0])
        with h_c1:
            h_name_esc = html.escape(h_name)
            j_esc = html.escape(jockey_now)
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:2px;">'
                f'<span style="font-size:1.12rem; font-weight:800; color:#0f172a;">{h_num}番 {h_name_esc}</span>'
                f'<span style="font-size:0.72rem; color:#2563eb; background:#eff6ff; border:1px solid #bfdbfe; '
                f'padding:1px 7px; border-radius:999px; font-weight:700;">KAZUYAN AI</span>'
                f'</div>'
                f'<div style="font-size:0.82rem; color:#64748b; margin-bottom:6px;">{j_esc} {kinryo_now:.1f}kg</div>',
                unsafe_allow_html=True,
            )
        with h_c2:
            if st.button("✕", key="btn_close_horse_panel", help="詳細パネルを閉じる", width="stretch"):
                st.session_state["selected_detail_horse"] = None
                try:
                    if "horse" in st.query_params:
                        del st.query_params["horse"]
                except Exception:
                    pass
                st.rerun()

        # 2. 前後の馬ナビゲーション
        nav1, nav2, nav3 = st.columns([1.6, 0.9, 1.6])
        with nav1:
            if st.button("‹ 前の馬", key="btn_panel_prev_horse", disabled=(cur_idx <= 0), width="stretch"):
                prev_h = int(ordered_df.iloc[cur_idx - 1]["horse_num"])
                st.session_state["selected_detail_horse"] = prev_h
                st.rerun()
        with nav2:
            st.markdown(
                f'<div style="text-align:center; font-weight:800; color:#1e293b; padding-top:8px; font-size:0.9rem;">{cur_idx + 1} / {total_n}</div>',
                unsafe_allow_html=True,
            )
        with nav3:
            if st.button(
                "次の馬 ›",
                key="btn_panel_next_horse",
                type="primary",
                disabled=(cur_idx >= total_n - 1),
                width="stretch",
            ):
                next_h = int(ordered_df.iloc[cur_idx + 1]["horse_num"])
                st.session_state["selected_detail_horse"] = next_h
                st.rerun()

        # 3. バッジ行
        badges = [
            f'<span style="background:#dc2626; color:#fff; font-weight:800; font-size:0.74rem; padding:3px 9px; border-radius:5px;">AI{cur_idx + 1}位</span>',
            f'<span style="background:{exp["sig_color"]}; color:#fff; font-weight:800; font-size:0.74rem; padding:3px 10px; border-radius:5px;">{html.escape(exp["sig_label"])}</span>',
        ]
        if exp["conf_label"]:
            badges.append(
                f'<span style="background:{exp["conf_color"]}; color:#fff; font-weight:800; font-size:0.74rem; padding:3px 9px; border-radius:5px;">{html.escape(exp["conf_label"])}</span>'
            )
        ev_badge_bg = "#f59e0b" if ev_now >= ev_threshold else "#475569"
        badges.append(
            f'<span style="background:{ev_badge_bg}; color:#fff; font-weight:800; font-size:0.74rem; padding:3px 9px; border-radius:5px;">単勝EV {ev_now:.2f}</span>'
        )
        badges_html = "".join(badges)

        st.markdown(
            f'<div style="display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 8px 0;">{badges_html}</div>'
            f'<div style="display:flex; justify-content:space-between; align-items:center; background:#f8fafc; '
            f'border:1px solid #e2e8f0; border-radius:7px; padding:7px 11px; margin-bottom:8px; font-size:0.84rem;">'
            f'<span><strong>{pop_now}番人気</strong> ／ 単勝 <strong>{odds_now:.1f}倍</strong></span>'
            f'<span style="color:#dc2626; font-weight:800;">能力値 {float(row["ability_score"]):.1f}</span>'
            f'</div>'
            f'<div style="font-size:0.86rem; color:#0f172a; font-weight:700; margin-bottom:8px;">'
            f'勝率 {win_p:.1f}%、2着内率 {quin_p:.1f}%、3着内率 {place_p:.1f}%。</div>',
            unsafe_allow_html=True,
        )

        # 4. シグナル判定理由 ＆ AI評価根拠
        contribs = analyze_feature_contributions(
            race_df=ordered_df,
            target_horse_num=h_num,
            model=model,
            feature_cols=feature_cols,
            category_maps=category_maps,
            train_model=train_model,
        )
        factor_pills = []
        for c_item in contribs[:4]:
            dp = float(c_item["delta_pct"])
            if dp >= 0.1:
                c_bg, c_fg = "#fef2f2", "#dc2626"
            elif dp <= -0.1:
                c_bg, c_fg = "#eff6ff", "#1d4ed8"
            else:
                c_bg, c_fg = "#f1f5f9", "#475569"
            f_esc = html.escape(c_item["factor"])
            d_esc = html.escape(c_item["detail"])
            factor_pills.append(
                f'<div style="display:flex; justify-content:space-between; font-size:0.78rem; padding:3px 0; border-bottom:1px dashed #e2e8f0;">'
                f'<span style="color:#334155;"><strong>{f_esc}</strong>（{d_esc}）</span>'
                f'<span style="background:{c_bg}; color:{c_fg}; font-weight:800; padding:0 6px; border-radius:4px;">{dp:+.2f}%</span>'
                f'</div>'
            )
        factors_body = "".join(factor_pills)
        reason_esc = html.escape(exp["main_reason"])
        gap_txt = f"{exp['gap_pct']:+.1f}pt"

        st.markdown(
            f'<div style="background:{exp["sig_bg"]}; border-left:4px solid {exp["sig_color"]}; border-radius:6px; '
            f'padding:9px 11px; margin-bottom:12px;">'
            f'<div style="font-size:0.78rem; font-weight:800; color:{exp["sig_color"]}; margin-bottom:3px;">なぜ「{html.escape(exp["sig_label"])}」？</div>'
            f'<div style="font-size:0.81rem; color:#1e293b; line-height:1.45; margin-bottom:6px;">{reason_esc}</div>'
            f'<div style="font-size:0.75rem; color:#475569; margin-bottom:4px;">市場評価 {exp["mkt_implied_pct"]:.1f}% vs AI勝率 {win_p:.1f}%（乖離 {gap_txt}）</div>'
            f'{factors_body}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 5. Webから基本プロフィール＆全戦績取得
        with st.spinner("基本データ・戦績傾向を取得中..."):
            prof = fetch_horse_profile_and_past5(race_id=str(race_id), horse_num=h_num, horse_name=h_name)

        runs = prof.get("runs", [])

        # 6. 特徴・適性サマリー（コース適性・季節・馬場・脚質）
        tendency = summarize_horse_tendencies(
            all_runs=runs,
            target_venue=venue_now,
            target_surface=surf_now,
            target_distance=dist_now,
            target_date=date_str,
            target_cond=cond_now,
        )
        t_head_esc = html.escape(tendency["headline"])
        t_badges_html = "".join(
            f'<span style="background:{bg_c}; color:{fg_c}; border:1px solid {fg_c}33; '
            f'font-size:0.72rem; font-weight:800; padding:2px 8px; border-radius:999px;">{html.escape(lbl)}</span>'
            for lbl, fg_c, bg_c in tendency["badges"]
        )
        t_bullets_html = "".join(
            f'<li style="margin-bottom:2px;">{html.escape(b)}</li>'
            for b in tendency["bullets"]
        )
        badges_row_div = (
            f'<div style="display:flex; gap:5px; flex-wrap:wrap; margin:5px 0 6px 0;">{t_badges_html}</div>'
            if t_badges_html else ""
        )
        bullets_ul = (
            f'<ul style="margin:4px 0 0 0; padding-left:17px; font-size:0.77rem; color:#334155; line-height:1.45;">{t_bullets_html}</ul>'
            if t_bullets_html else ""
        )
        st.markdown(
            '<div style="font-size:0.84rem; font-weight:800; color:#192442; border-left:3.5px solid #047857; padding-left:7px; margin:6px 0 6px 0;">特徴・適性サマリー</div>'
            '<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:7px; padding:9px 11px; margin-bottom:12px;">'
            f'<div style="font-size:0.82rem; font-weight:800; color:#0f172a; line-height:1.45;">{t_head_esc}</div>'
            f'{badges_row_div}'
            f'{bullets_ul}'
            '</div>',
            unsafe_allow_html=True,
        )

        # 7. 基本プロフィール
        sex_age = prof.get("sex_age") or "-"
        trainer_disp = prof.get("trainer") or trainer_now
        sire = prof.get("sire") or ""
        bms = prof.get("bms") or ""
        blood_disp = f"父 {sire}・母父 {bms}" if (sire and bms) else (f"父 {sire}" if sire else "-")

        if runs:
            r0 = runs[0]
            prev_rk_str = str(r0.get("rank", "-"))
            prev_mark = "○ " if prev_rk_str.isdigit() and int(prev_rk_str) <= 3 else ""
            prev_disp = f"{prev_mark}{prev_rk_str}着（{r0.get('venue', '')}{r0.get('distance', '')}）" if prev_rk_str.isdigit() else prev_rk_str

            cur_dt = _parse_date_obj(date_str.replace("-", "/"))
            prev_dt = _parse_date_obj(r0.get("date_full", ""))
            if cur_dt and prev_dt and (cur_dt - prev_dt).days > 0:
                days_gap = (cur_dt - prev_dt).days
                weeks_gap = max(0, (days_gap - 1) // 7)
                gap_disp = f"中{weeks_gap}週（中{days_gap}日）"
            else:
                gap_disp = "-"

            prev_jk = str(r0.get("jockey", "") or "").strip()
            if prev_jk and prev_jk != "-":
                if _norm_jockey_key(prev_jk) == _norm_jockey_key(jockey_now):
                    jk_disp = f"○ 継続騎乗（{jockey_now}）"
                else:
                    jk_disp = f"△ 乗り替わり（前走 {prev_jk}）"
            else:
                jk_disp = jockey_now

            num_ranks = [int(r["rank"]) for r in runs[:5] if str(r.get("rank", "")).isdigit()]
            if num_ranks:
                w_cnt = sum(1 for x in num_ranks if x == 1)
                p_cnt = sum(1 for x in num_ranks if x <= 3)
                w_rate = int(round(w_cnt / len(num_ranks) * 100))
                p_rate = int(round(p_cnt / len(num_ranks) * 100))
                recent_disp = f"直近{len(num_ranks)}走 勝率{w_rate}% / 複勝率{p_rate}%"
            else:
                recent_disp = "-"
        else:
            p1_fb = int(row.get("prev1_rank", 7))
            prev_disp = f"{p1_fb}着（出馬表データ）"
            gap_disp = "-"
            jk_disp = jockey_now
            recent_disp = "-"

        career_disp = prof.get("career") or "-"

        basic_rows = [
            ("性齢", sex_age),
            ("厩舎", trainer_disp),
            ("血統", blood_disp),
            ("前走", prev_disp),
            ("間隔", gap_disp),
            ("騎手", jk_disp),
            ("近走", recent_disp),
            ("通算", career_disp),
        ]
        basic_html_rows = "".join(
            f'<tr>'
            f'<td style="color:#64748b; width:46px; padding:3px 4px; font-weight:600; vertical-align:top;">{k}</td>'
            f'<td style="color:#0f172a; padding:3px 4px; font-weight:600;">{html.escape(str(v))}</td>'
            f'</tr>'
            for k, v in basic_rows
        )
        st.markdown(
            '<div style="font-size:0.84rem; font-weight:800; color:#192442; border-left:3.5px solid #4a154b; padding-left:7px; margin:8px 0 6px 0;">基本</div>'
            f'<table style="width:100%; border-collapse:collapse; font-size:0.79rem; margin-bottom:12px;">{basic_html_rows}</table>',
            unsafe_allow_html=True,
        )

        # 8. 近5走セクション
        st.markdown(
            '<div style="font-size:0.84rem; font-weight:800; color:#192442; border-left:3.5px solid #2563eb; padding-left:7px; margin:8px 0 6px 0;">近5走</div>',
            unsafe_allow_html=True,
        )

        if runs:
            ref_y = str(date_str)[:4]
            tr_parts = []
            for r in runs[:5]:
                d_s = html.escape(_short_date(r.get("date_full", ""), ref_year=ref_y))
                v_s = html.escape(str(r.get("venue", "-")))
                dist_s = html.escape(str(r.get("distance", "-")))
                cond_s = html.escape(str(r.get("condition", "-")))
                rk_raw = str(r.get("rank", "-"))
                hd_raw = str(r.get("head", ""))
                rk_disp = f"{rk_raw}/{hd_raw}" if (rk_raw.isdigit() and hd_raw) else rk_raw
                if rk_raw == "1":
                    rk_style = "color:#dc2626; font-weight:900;"
                elif rk_raw == "2":
                    rk_style = "color:#ea580c; font-weight:800;"
                elif rk_raw == "3":
                    rk_style = "color:#2563eb; font-weight:800;"
                else:
                    rk_style = "color:#334155; font-weight:600;"
                pop_s = html.escape(str(r.get("popularity", "-") or "-"))
                tm_s = html.escape(str(r.get("time", "-") or "-"))
                ag_s = html.escape(str(r.get("agari", "-") or "-"))
                wt_s = html.escape(str(r.get("weight", "-") or "-"))

                tr_parts.append(
                    '<tr style="border-bottom:1px solid #f1f5f9;">'
                    f'<td style="padding:4px 2px; color:#475569;">{d_s}</td>'
                    f'<td style="padding:4px 2px; color:#0f172a; font-weight:600;">{v_s}</td>'
                    f'<td style="padding:4px 2px; color:#0f172a;">{dist_s}</td>'
                    f'<td style="padding:4px 2px; color:#475569;">{cond_s}</td>'
                    f'<td style="padding:4px 2px; {rk_style}">{html.escape(rk_disp)}</td>'
                    f'<td style="padding:4px 2px; color:#475569;">{pop_s}</td>'
                    f'<td style="padding:4px 2px; color:#0f172a;">{tm_s}</td>'
                    f'<td style="padding:4px 2px; color:#0f172a;">{ag_s}</td>'
                    f'<td style="padding:4px 2px; color:#475569; font-size:0.71rem;">{wt_s}</td>'
                    '</tr>'
                )
            past_tbl_html = (
                '<div style="overflow-x:auto; margin-bottom:10px;">'
                '<table style="width:100%; border-collapse:collapse; font-size:0.74rem; text-align:center; white-space:nowrap;">'
                '<thead><tr style="border-bottom:1.5px solid #cbd5e1; color:#64748b; font-size:0.71rem;">'
                '<th style="padding:4px 2px;">日付</th><th style="padding:4px 2px;">場</th>'
                '<th style="padding:4px 2px;">距離</th><th style="padding:4px 2px;">馬場</th>'
                '<th style="padding:4px 2px;">着</th><th style="padding:4px 2px;">人気</th>'
                '<th style="padding:4px 2px;">タイム</th><th style="padding:4px 2px;">上り</th>'
                '<th style="padding:4px 2px;">体重</th>'
                '</tr></thead>'
                f'<tbody>{"".join(tr_parts)}</tbody>'
                '</table></div>'
            )
            st.markdown(past_tbl_html, unsafe_allow_html=True)
        else:
            st.caption("過去5走の詳細データが見つかりませんでした（新馬・初出走、または取得制限の可能性があります）。")

        if prof.get("horse_url"):
            st.link_button("netkeibaで全戦績を見る", url=prof["horse_url"], width="stretch")