import html
import os
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_past5_details(race_id: str, horse_num: int) -> dict:
    """
    shutuba_past.html から指定馬番の過去5走の詳細成績と horse_id を取得する。
    """
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    result = {
        "horse_id": "",
        "horse_url": "",
        "runs": [],
        "error": "",
    }
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            result["error"] = f"過去走ページの取得に失敗しました (HTTP {resp.status_code})"
            return result

        soup = _make_soup(_decode_html(resp))
        target_tr = None
        for tr in soup.select("tr.HorseList"):
            num_el = tr.select_one("td.Waku + td") or tr.select_one("td[class*='Umaban']")
            if not num_el:
                continue
            ntxt = re.sub(r"\D", "", num_el.get_text(strip=True))
            if ntxt.isdigit() and int(ntxt) == int(horse_num):
                target_tr = tr
                break

        if target_tr is None:
            result["error"] = "出馬表の過去走データに対象馬が見つかりませんでした。"
            return result

        # 馬の個別ページURL取得
        h_link = target_tr.select_one("div.Horse02 a[href*='/horse/']") or target_tr.select_one("a[href*='/horse/']")
        if h_link and h_link.get("href"):
            href = h_link["href"]
            hm = re.search(r"/horse/([a-zA-Z0-9]+)", href)
            if hm:
                result["horse_id"] = hm.group(1)
                result["horse_url"] = f"https://db.netkeiba.com/horse/{hm.group(1)}/"

        runs = []
        past_tds = target_tr.select("td.Past")
        for idx_p, ptd in enumerate(past_tds[:5], start=1):
            full_txt = re.sub(r"\s+", " ", ptd.get_text(" ", strip=True))
            if not full_txt or full_txt in ("-", "--"):
                continue

            # Data01: 日付・開催・着順
            d01 = ptd.select_one("div.Data01")
            rank_el = ptd.select_one("span.Num")
            rank_str = rank_el.get_text(strip=True) if rank_el else "-"

            d01_txt = re.sub(r"\s+", " ", d01.get_text(" ", strip=True)) if d01 else full_txt
            m_date = re.search(r"(\d{4}\.\d{2}\.\d{2})", d01_txt)
            date_val = m_date.group(1) if m_date else "-"

            venue_val = "-"
            if d01:
                v_clean = re.sub(r"\d{4}\.\d{2}\.\d{2}", "", d01_txt)
                if rank_str != "-":
                    v_clean = re.sub(rf"\b{re.escape(rank_str)}\b", "", v_clean)
                v_clean = v_clean.strip()
                if v_clean:
                    venue_val = v_clean.split()[0]

            # Data02: レース名
            d02 = ptd.select_one("div.Data02")
            race_name = re.sub(r"\s+", " ", d02.get_text(" ", strip=True)) if d02 else "-"

            # Data03: 頭数・馬番・人気・騎手・斤量
            d03 = ptd.select_one("div.Data03")
            d03_txt = re.sub(r"\s+", " ", d03.get_text(" ", strip=True)) if d03 else ""
            m_pop = re.search(r"(\d+)人", d03_txt)
            pop_val = f"{m_pop.group(1)}人気" if m_pop else "-"
            m_head = re.search(r"(\d+)頭", d03_txt)
            head_val = f"{m_head.group(1)}頭" if m_head else "-"

            jockey_kinryo = re.sub(r"\d+頭|\d+番|\d+人", "", d03_txt).strip()
            jockey_kinryo = re.sub(r"\s+", " ", jockey_kinryo) or "-"

            # Data05: コース・距離・タイム・馬場
            d05 = ptd.select_one("div.Data05")
            d05_txt = re.sub(r"\s+", " ", d05.get_text(" ", strip=True)) if d05 else ""
            m_course = re.search(r"([芝ダ障]\d{3,4})", d05_txt)
            course_val = m_course.group(1) if m_course else "-"
            m_time = re.search(r"(\d+:\d{2}\.\d|\d{2}\.\d)", d05_txt)
            time_val = m_time.group(1) if m_time else "-"
            m_cond = re.search(r"(良|稍重|稍|重|不良)", d05_txt)
            cond_val = m_cond.group(1) if m_cond else "-"

            # Data06: 通過順・上がり3F・馬体重
            d06 = ptd.select_one("div.Data06")
            d06_txt = re.sub(r"\s+", " ", d06.get_text(" ", strip=True)) if d06 else ""
            m_agari = re.search(r"\((\d{2}\.\d)\)", d06_txt)
            agari_val = m_agari.group(1) if m_agari else "-"
            m_wt = re.search(r"(\d{3}\([+-]?\d+\))", d06_txt)
            wt_val = m_wt.group(1) if m_wt else "-"
            m_pass = re.search(r"^([\d-]+)", d06_txt)
            pass_val = m_pass.group(1) if m_pass else "-"

            # Data07: 相手馬・着差
            d07 = ptd.select_one("div.Data07")
            d07_txt = re.sub(r"\s+", " ", d07.get_text(" ", strip=True)) if d07 else ""
            m_diff = re.search(r"\(([+-]?\d+\.\d)\)", d07_txt)
            diff_val = f"{m_diff.group(1)}秒" if m_diff else "-"
            rival_val = re.sub(r"\([+-]?\d+\.\d\)", "", d07_txt).strip() or "-"

            runs.append({
                "走前": f"{idx_p}走前",
                "日付": date_val,
                "開催": venue_val,
                "レース名": race_name,
                "条件": f"{course_val} ({cond_val})" if course_val != "-" else "-",
                "頭数": head_val,
                "人気": pop_val,
                "着順": f"{rank_str}着" if rank_str.isdigit() else rank_str,
                "タイム": time_val,
                "着差": diff_val,
                "通過": pass_val,
                "上がり3F": agari_val,
                "騎手・斤量": jockey_kinryo,
                "馬体重": wt_val,
                "勝ち馬(2着馬)": rival_val,
            })

        result["runs"] = runs
    except Exception as exc:
        result["error"] = f"過去走データの取得中にエラーが発生しました: {exc}"

    return result


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
    feat_df, _ = train_model.create_features(race_df, category_maps=category_maps)
    base_raw = np.clip(model.predict_proba(feat_df[feature_cols])[:, 1], 1e-6, 1.0)
    base_probs = base_raw / base_raw.sum()

    matches = race_df.index[race_df["horse_num"].astype(int) == int(target_horse_num)].tolist()
    if not matches:
        return []
    pos = race_df.index.get_loc(matches[0])
    target_base_prob = float(base_probs[pos])

    row_src = race_df.iloc[pos]
    groups = [
        {
            "name": "近3走の着順・勢い",
            "cols": ["prev1_rank", "prev2_rank", "prev3_rank", "avg_prev_rank", "best_prev_rank", "rank_trend", "rel_prev_rank"],
            "desc": f"前走{int(row_src.get('prev1_rank', 7))}着 / 2走前{int(row_src.get('prev2_rank', 7))}着 / 3走前{int(row_src.get('prev3_rank', 7))}着",
        },
        {
            "name": "前走タイム差（内容の濃さ）",
            "cols": ["prev1_time_diff", "rel_time_diff"],
            "desc": f"前走勝ち馬との差 {float(row_src.get('prev1_time_diff', 0.8)):.2f}秒",
        },
        {
            "name": "騎手の実績・コース相性",
            "cols": ["jockey_id"],
            "desc": f"騎乗: {row_src.get('jockey_id', '-')}",
        },
        {
            "name": "調教師（厩舎力）",
            "cols": ["trainer_id"],
            "desc": f"厩舎: {row_src.get('trainer_id', '-')}",
        },
        {
            "name": "枠順・馬番の有利不利",
            "cols": ["frame_num", "horse_num"],
            "desc": f"{int(row_src.get('frame_num', 1))}枠 {int(row_src.get('horse_num', 1))}番",
        },
        {
            "name": "斤量・馬体重バランス",
            "cols": ["weight_carried", "horse_weight", "weight_diff", "abs_weight_diff", "weight_ratio"],
            "desc": f"斤量{float(row_src.get('weight_carried', 56.0)):.1f}kg / 馬体重{int(row_src.get('horse_weight', 476))}kg ({int(row_src.get('weight_diff', 0)):+d})",
        },
    ]

    contribs = []
    for g in groups:
        mod_feat = feat_df[feature_cols].copy()
        valid_cols = [c for c in g["cols"] if c in mod_feat.columns]
        if not valid_cols:
            continue

        for c in valid_cols:
            if isinstance(mod_feat[c].dtype, pd.CategoricalDtype):
                mode_vals = mod_feat[c].mode()
                if len(mode_vals) > 0:
                    mod_feat.iloc[pos, mod_feat.columns.get_loc(c)] = mode_vals.iloc[0]
            else:
                med_val = float(mod_feat[c].median())
                mod_feat.iloc[pos, mod_feat.columns.get_loc(c)] = med_val

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
    """
    シグナル（軸・穴注目・紐候補・様子見）および信頼度（S/A/B/-）の判定理由を具体的に文章化する。
    """
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

    # シグナル判定（03_app.py と完全一致）
    if ai_rank == 1 and not (exclude_fav1 and pop == 1):
        sig_label = "軸"
        sig_color = "#dc2626"
        sig_bg = "#fef2f2"
        main_reason = (
            f"出走{len(race_df)}頭の中でAI予測勝率が1位（{win_p:.1f}% / 能力値 {ability:.1f}）であり、"
            f"3着内率も{place_p:.1f}%と高水準なため、馬券の中心となる『軸』判定となりました。"
        )
    elif ev >= ev_threshold and win_p >= 4.0:
        sig_label = "穴注目"
        sig_color = "#c2410c"
        sig_bg = "#fff7ed"
        main_reason = (
            f"単勝オッズ{odds:.1f}倍（{pop}番人気）の市場評価（想定勝率 {mkt_implied_pct:.1f}%）に対し、"
            f"AIは勝率 {win_p:.1f}% と高く評価しています。単勝期待値(EV)が {ev:.2f} と基準値（{ev_threshold:.2f}）を"
            f"上回り、勝率も最低ライン（4.0%）を超えているため、妙味抜群の『穴注目』判定となりました。"
        )
    elif ai_rank <= 6 or place_p >= 25.0:
        sig_label = "紐候補"
        sig_color = "#1d4ed8"
        sig_bg = "#eff6ff"
        cond_parts = []
        if ai_rank <= 6:
            cond_parts.append(f"AI総合順位が上位6頭以内（{ai_rank}位）")
        if place_p >= 25.0:
            cond_parts.append(f"3着内率が25%以上（{place_p:.1f}%）")
        joined = "かつ".join(cond_parts)
        if ai_rank == 1 and exclude_fav1 and pop == 1:
            main_reason = (
                f"AI順位は1位（勝率 {win_p:.1f}%）ですが、「1番人気を軸から除外」設定がONのため軸指定を見送り、"
                f"相手筆頭の『紐候補』としています。"
            )
        else:
            main_reason = (
                f"{joined}を満たしているため、2・3着の連下・三連複の相手として押さえるべき『紐候補』判定となりました。"
                f"（単勝EVは {ev:.2f} のため、単勝狙いより連系馬券の相手向きです）"
            )
    else:
        sig_label = "様子見"
        sig_color = "#475569"
        sig_bg = "#f8fafc"
        main_reason = (
            f"AI順位が{ai_rank}位（勝率 {win_p:.1f}% / 3着内率 {place_p:.1f}%）、単勝EVも {ev:.2f} に留まるため、"
            f"現時点のオッズでは積極的な投資妙味が薄い『様子見』判定です。"
        )

    # 信頼度理由
    if ai_rank == 1 and win_p >= 22.0:
        conf_label = "S"
        conf_reason = f"AI1位かつ予測勝率が22%超（{win_p:.1f}%）の抜けた評価のため、最高ランク『S』です。"
    elif ai_rank == 1 or (ai_rank == 2 and win_p >= 14.0):
        conf_label = "A"
        conf_reason = f"AI上位（{ai_rank}位）かつ高い勝率（{win_p:.1f}%）を維持しているため、有力ランク『A』です。"
    elif ai_rank <= 3 and win_p >= 9.0:
        conf_label = "B"
        conf_reason = f"AIトップ3（{ai_rank}位）かつ勝率9%以上（{win_p:.1f}%）のため、標準上位ランク『B』です。"
    else:
        conf_label = "-"
        conf_reason = f"上位抜擢基準（AI3位以内＆勝率9%以上）を下回るため、信頼度ランクは付与されていません。"

    # 箇条書きポイント
    bullets = []
    if gap_pct >= 2.0:
        bullets.append(
            f"オッズとのギャップ（妙味）：市場評価 {mkt_implied_pct:.1f}% に対し AI予測 {win_p:.1f}%（+{gap_pct:.1f}pt 過小評価）"
        )
    elif gap_pct <= -2.0:
        bullets.append(
            f"オッズとのギャップ（過剰人気気味）：市場評価 {mkt_implied_pct:.1f}% に対し AI予測 {win_p:.1f}%（{gap_pct:.1f}pt 割高）"
        )
    else:
        bullets.append(
            f"オッズとの整合性：市場評価 {mkt_implied_pct:.1f}% と AI予測 {win_p:.1f}% が概ね一致しています"
        )

    p1 = int(row.get("prev1_rank", 7))
    p2 = int(row.get("prev2_rank", 7))
    p3 = int(row.get("prev3_rank", 7))
    avg_p = (p1 + p2 + p3) / 3.0
    race_avg_p = float(race_df[["prev1_rank", "prev2_rank", "prev3_rank"]].mean(axis=1).mean())
    if avg_p < race_avg_p - 0.5:
        bullets.append(f"直近3走の安定感：平均着順 {avg_p:.1f}着（出走馬平均 {race_avg_p:.1f}着より優秀）")
    elif p1 <= 3:
        bullets.append(f"前走好走の勢い：前走 {p1}着（2走前 {p2}着 / 3走前 {p3}着）")
    else:
        bullets.append(f"直近3走の推移：前走 {p1}着 → 2走前 {p2}着 → 3走前 {p3}着（出走馬平均 {race_avg_p:.1f}着）")

    tdiff = float(row.get("prev1_time_diff", 0.8))
    if tdiff <= 0.3:
        bullets.append(f"前走の着差が優秀：勝ち馬と {tdiff:.2f}秒差の接戦・好内容")
    elif tdiff >= 1.5:
        bullets.append(f"前走の着差に課題：勝ち馬から {tdiff:.2f}秒離されており、巻き返し条件が鍵")

    bullets.append(f"ハービル連対・複勝見込み：勝率 {win_p:.1f}% ／ 2着内率 {quin_p:.1f}% ／ 3着内率 {place_p:.1f}%")

    return {
        "sig_label": sig_label,
        "sig_color": sig_color,
        "sig_bg": sig_bg,
        "main_reason": main_reason,
        "conf_label": conf_label,
        "conf_reason": conf_reason,
        "bullets": bullets,
        "mkt_implied_pct": mkt_implied_pct,
        "gap_pct": gap_pct,
    }


def render_horse_detail_page(
    race_id: str,
    date_str: str,
    horse_num: int,
    ai_weight: float,
    ev_threshold: float,
    exclude_fav1: bool,
    model,
    feature_cols: list[str],
    category_maps: dict,
    train_model,
    shutuba_scraper,
    calculate_horse_place_probs,
):
    """
    別タブで開かれた馬詳細ページ（過去5走成績 ＋ シグナル判定理由 ＋ AI特徴量寄与度）を描画する。
    """
    st.markdown(
        """
<div class="top-navbar">
  <div class="brand">KAZUYAN AI <span style="font-size:0.85rem; font-weight:600; color:#cbd5e1; margin-left:8px;">個別馬AI診断・過去5走レポート</span></div>
  <div class="nav-badges">
    <a href="?" style="color:#ffffff; text-decoration:none; background:rgba(255,255,255,0.18); padding:5px 12px; border-radius:6px; font-weight:700;">← レース一覧へ戻る</a>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # レースデータの読み込み（まず同じrace_idの保存CSVを確認し、無ければWeb取得）
    race_df = None
    if os.path.exists("data/upcoming_race.csv"):
        try:
            cached_df = pd.read_csv("data/upcoming_race.csv", dtype={"race_id": str})
            if not cached_df.empty and str(cached_df["race_id"].iloc[0]) == str(race_id):
                race_df = cached_df
        except Exception:
            race_df = None

    if race_df is None:
        with st.spinner("対象レースの出馬表とオッズを取得中..."):
            try:
                bundle = shutuba_scraper.fetch_shutuba_and_odds(race_id=race_id, date_str=date_str)
                race_df = bundle["shutuba_df"].copy()
            except Exception as exc:
                st.error(f"レースデータの読み込みに失敗しました: {exc}")
                return

    # 推論・スコア計算（メイン画面と完全同一ロジック）
    feat_df, _ = train_model.create_features(race_df, category_maps=category_maps)
    raw_probs = model.predict_proba(feat_df[feature_cols])[:, 1]
    ai_pure_prob = raw_probs / raw_probs.sum()

    mkt_raw = 0.80 / np.clip(race_df["win_odds"].to_numpy(dtype=float), 1.1, 500.0)
    mkt_prob = mkt_raw / mkt_raw.sum()

    blended_prob = (ai_weight * ai_pure_prob) + ((1.0 - ai_weight) * mkt_prob)
    race_df["win_prob"] = blended_prob / blended_prob.sum()
    race_df["ai_pure_pct"] = ai_pure_prob * 100.0
    race_df = calculate_horse_place_probs(race_df, prob_col="win_prob")

    race_df["popularity"] = race_df["win_odds"].rank(method="min", ascending=True).astype(int)
    race_df["単勝期待値(EV)"] = (race_df["pred_win_pct"] / 100.0) * race_df["win_odds"]
    p_mean = race_df["pred_win_pct"].mean()
    p_std = max(race_df["pred_win_pct"].std(), 1e-6)
    race_df["ability_score"] = np.clip(50.0 + ((race_df["pred_win_pct"] - p_mean) / p_std) * 8.5, 35.0, 78.0).round(1)

    race_df = race_df.sort_values("pred_win_pct", ascending=False).reset_index(drop=True)
    race_df["ai_rank"] = np.arange(1, len(race_df) + 1)
    marks = ["◎", "○", "▲", "△", "☆"]
    race_df["印"] = [marks[i] if i < len(marks) else "--" for i in range(len(race_df))]

    target_rows = race_df[race_df["horse_num"].astype(int) == int(horse_num)]
    if target_rows.empty:
        st.error(f"馬番 {horse_num} のデータが見つかりませんでした。")
        return

    row = target_rows.iloc[0]
    exp = build_signal_explanation(row, race_df, ev_threshold=ev_threshold, exclude_fav1=exclude_fav1)

    # 馬プロフィールヘッダーバナー
    waku = int(np.clip(row["frame_num"], 1, 8))
    h_name = html.escape(str(row["horse_name"]))
    jockey = html.escape(str(row["jockey_id"]))
    trainer = html.escape(str(row.get("trainer_id", "-")))
    venue = html.escape(str(row.get("venue", "")))
    r_num = int(str(race_id)[-2:]) if len(str(race_id)) >= 2 and str(race_id)[-2:].isdigit() else 11
    r_title = html.escape(str(row.get("race_title", f"{venue}{r_num}R")))
    surf = "ダ" if "ダ" in str(row.get("surface", "芝")) else "芝"
    dist = int(float(row.get("distance", 2000)))
    cond = html.escape(str(row.get("track_condition", "良")))

    header_html = (
        f'<div class="race-banner">'
        f'<div>'
        f'<div class="race-title-row">'
        f'<span class="waku-badge waku-{waku}">{waku}</span>'
        f'<span class="race-venue-r">{int(horse_num)}番 {h_name}</span>'
        f'<span class="race-time">印: {html.escape(str(row["印"]))} ／ AI {int(row["ai_rank"])}位</span>'
        f'</div>'
        f'<div class="race-meta-row" style="margin-top:8px;">'
        f'<span>{venue} {r_num}R {r_title}（{surf}{dist}m・{cond}）</span>'
        f'<span>騎手: {jockey}（{float(row["weight_carried"]):.1f}kg）</span>'
        f'<span>厩舎: {trainer}</span>'
        f'<span>単勝: {float(row["win_odds"]):.1f}倍（{int(row["popularity"])}人気）</span>'
        f'</div>'
        f'</div>'
        f'<div class="ring-group">'
        f'<div class="ring-circle ring-orange">'
        f'<span class="ring-num-orange">{float(row["単勝期待値(EV)"]):.2f}</span>'
        f'<span class="ring-label">単勝EV</span>'
        f'</div>'
        f'<div class="ring-circle ring-green">'
        f'<span class="ring-num-green">{float(row["ability_score"]):.1f}</span>'
        f'<span class="ring-label">能力値</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # KPI指標
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("AI予測勝率", f"{float(row['pred_win_pct']):.1f}%", f"純粋AI: {float(row['ai_pure_pct']):.1f}%")
    k2.metric("2着内率（連対）", f"{float(row['pred_quinella_pct']):.1f}%")
    k3.metric("3着内率（複勝）", f"{float(row['pred_place_pct']):.1f}%")
    k4.metric("市場想定勝率", f"{exp['mkt_implied_pct']:.1f}%", f"乖離: {exp['gap_pct']:+.1f}pt")
    k5.metric("単勝期待値 (EV)", f"{float(row['単勝期待値(EV)']):.2f}", f"基準: {ev_threshold:.2f}")

    # ==========================================================================
    # セクション1：なぜこのシグナルなのか？（判定理由 ＆ AI評価根拠）
    # ==========================================================================
    st.markdown("### 1. シグナル判定理由・AI評価の根拠")

    bullets_html = "".join(f"<li style='margin-bottom:5px;'>{html.escape(b)}</li>" for b in exp["bullets"])
    sig_card_html = (
        f'<div style="background:{exp["sig_bg"]}; border:1px solid #cbd5e1; border-left:6px solid {exp["sig_color"]}; '
        f'border-radius:10px; padding:16px 20px; margin-bottom:16px;">'
        f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">'
        f'<span style="background:{exp["sig_color"]}; color:#ffffff; font-weight:800; padding:4px 14px; border-radius:6px; font-size:0.95rem;">シグナル：{exp["sig_label"]}</span>'
        f'<span style="font-weight:800; color:#192442; font-size:0.95rem;">信頼度ランク：{exp["conf_label"]}</span>'
        f'</div>'
        f'<p style="color:#0f172a; font-weight:700; font-size:0.96rem; line-height:1.6; margin-bottom:8px;">{html.escape(exp["main_reason"])}</p>'
        f'<p style="color:#475569; font-size:0.86rem; margin-bottom:10px;"><strong>信頼度の根拠：</strong>{html.escape(exp["conf_reason"])}</p>'
        f'<ul style="margin:0; padding-left:20px; color:#1e293b; font-size:0.88rem;">{bullets_html}</ul>'
        f'</div>'
    )
    st.markdown(sig_card_html, unsafe_allow_html=True)

    # AIモデルの特徴量別プラス・マイナス寄与度
    with st.spinner("AIモデルの評価内訳を計算中..."):
        contribs = analyze_feature_contributions(
            race_df=race_df,
            target_horse_num=horse_num,
            model=model,
            feature_cols=feature_cols,
            category_maps=category_maps,
            train_model=train_model,
        )

    if contribs:
        st.markdown("#### AIモデルの評価内訳（同レース出走馬の平均と比べた勝率への影響度）")
        c_rows = []
        for c_item in contribs:
            dp = float(c_item["delta_pct"])
            if dp >= 0.5:
                judge = "強いプラス要因"
            elif dp > 0.05:
                judge = "プラス要因"
            elif dp <= -0.5:
                judge = "マイナス要因"
            elif dp < -0.05:
                judge = "ややマイナス"
            else:
                judge = "平均並み"
            c_rows.append({
                "評価ファクター": c_item["factor"],
                "この馬のデータ": c_item["detail"],
                "勝率への寄与": f"{dp:+.2f}%",
                "AI判定": judge,
            })
        st.dataframe(pd.DataFrame(c_rows), width="stretch", hide_index=True)

    # ==========================================================================
    # セクション2：過去5走の成績詳細
    # ==========================================================================
    st.markdown("### 2. 過去5走の成績詳細")
    with st.spinner("過去5走の詳細成績を取得中..."):
        past_info = fetch_past5_details(race_id=str(race_id), horse_num=int(horse_num))

    if past_info["runs"]:
        runs_df = pd.DataFrame(past_info["runs"])
        st.dataframe(runs_df, width="stretch", hide_index=True)
    else:
        st.warning(past_info["error"] or "過去5走の詳細データを取得できませんでした（新馬戦・初出走、またはデータ未公開の可能性があります）。")
        fallback_df = pd.DataFrame([
            {"走前": "前走", "着順": f"{int(row.get('prev1_rank', 7))}着", "備考": f"前走タイム差 {float(row.get('prev1_time_diff', 0.8)):.2f}秒"},
            {"走前": "2走前", "着順": f"{int(row.get('prev2_rank', 7))}着", "備考": "出馬表サマリーより"},
            {"走前": "3走前", "着順": f"{int(row.get('prev3_rank', 7))}着", "備考": "出馬表サマリーより"},
        ])
        st.dataframe(fallback_df, width="stretch", hide_index=True)

    if past_info.get("horse_url"):
        st.link_button("netkeiba でこの馬の全戦績・血統を見る", url=past_info["horse_url"])