from datetime import date, timedelta
import html
import itertools
import math
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


def get_default_win5_date() -> date:
    """土日WIN5対応：当日が土曜・日曜なら当日、平日（月〜金）なら次の土曜日を返す"""
    today = date.today()
    if today.weekday() in (5, 6):
        return today
    days_until_sat = (5 - today.weekday()) % 7
    return today + timedelta(days=days_until_sat)


def fetch_win5_target_races(date_str: str, shutuba_scraper) -> tuple[list[dict], list[dict], str]:
    """
    指定日のWIN5対象5レースを取得する。
    戻り値: (対象5レースリスト, 当日全レースリスト, 取得ソース説明)
    """
    ymd = date_str.replace("-", "")
    all_races = []
    try:
        all_races = shutuba_scraper.fetch_race_list_by_date(date_str)
    except Exception:
        all_races = []

    rid_to_item = {r["race_id"]: r for r in all_races}

    # 1. netkeiba WIN5専用ページから対象5レースIDを直接取得
    win5_rids = []
    for url in (
        f"https://race.netkeiba.com/top/win5.html?date={ymd}",
        "https://race.netkeiba.com/top/win5.html",
    ):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue
            html_txt = _decode_html(resp)
            y_i, m_i, d_i = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
            date_markers = (f"{y_i}年{m_i}月{d_i}日", f"{y_i}年{m_i:02d}月{d_i:02d}日", ymd)
            if "date=" not in url and not any(dm in html_txt for dm in date_markers):
                continue

            soup = _make_soup(html_txt)
            found = []
            for a_tag in soup.select('a[href*="race_id="]'):
                m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
                if not m:
                    continue
                rid = m.group(1)
                if rid[:4] != ymd[:4] or rid[4:6] not in JRA_VENUE_CODE:
                    continue
                if rid not in found:
                    found.append(rid)
                if len(found) == 5:
                    break
            if len(found) == 5:
                win5_rids = found
                break
        except Exception:
            continue

    if len(win5_rids) == 5:
        targets = []
        for idx, rid in enumerate(win5_rids, start=1):
            if rid in rid_to_item:
                item = dict(rid_to_item[rid])
            else:
                v_name = JRA_VENUE_CODE.get(rid[4:6], "東京")
                r_num = int(rid[10:12])
                item = {
                    "race_id": rid,
                    "venue": v_name,
                    "race_num": r_num,
                    "race_title": f"{v_name}{r_num}R",
                    "label": f"{v_name} {r_num:02d}R",
                }
            item["win5_leg"] = idx
            targets.append(item)
        return targets, all_races, "WIN5公式対象レース自動取得"

    # 2. フォールバック：当日全レースから10R・11Rの後半5レースを自動推定
    if all_races:
        candidates = [r for r in all_races if r["race_num"] in (10, 11)]
        if len(candidates) >= 5:
            r10 = sorted([r for r in candidates if r["race_num"] == 10], key=lambda x: x["race_id"][4:6], reverse=True)
            r11 = sorted([r for r in candidates if r["race_num"] == 11], key=lambda x: x["race_id"][4:6], reverse=True)
            if len(r11) >= 3:
                picked = r10[:2] + r11[:3]
            else:
                r09 = sorted([r for r in all_races if r["race_num"] == 9], key=lambda x: x["race_id"][4:6], reverse=True)
                picked = (r09[:1] + r10[:2] + r11[:2])[:5]
        else:
            picked = sorted(all_races, key=lambda x: (x["race_num"], x["race_id"]))[-5:]

        targets = []
        for idx, item in enumerate(picked[:5], start=1):
            cp = dict(item)
            cp["win5_leg"] = idx
            targets.append(cp)
        return targets, all_races, "メイン5レース自動推定（必要に応じて下で手動変更可）"

    return [], [], "指定日の開催レースが見つかりませんでした"


def compute_race_entropy(probs: np.ndarray) -> float:
    """勝率分布の正規化シャノンエントロピー（0=1強, 1=完全均等の大混戦）"""
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / math.log(max(len(p), 2)))


def classify_race_chaos(top1_p: float, top_gap: float, entropy: float) -> tuple[str, str]:
    """AI1位の勝率・2位との差・エントロピーからレースの混戦度を3段階で診断する"""
    if top1_p >= 0.25 or (top1_p >= 0.22 and top_gap >= 0.08) or entropy < 0.80:
        return "堅め（1点突破向き）", "#047857"
    if top1_p >= 0.195 or entropy < 0.87:
        return "標準（1〜2頭向き）", "#b45309"
    return "波乱・混戦（複数推奨）", "#dc2626"


def evaluate_win5_races(
    target_races: list[dict],
    date_str: str,
    model,
    feature_cols: list[str],
    category_maps: dict,
    train_model,
    shutuba_scraper,
    ai_weight: float,
) -> list[dict]:
    """WIN5対象5レースの出馬表・実オッズを一括取得し、AI勝率・市場支持率・混戦度を算出する"""
    evaluated = []
    for idx, r_item in enumerate(target_races, start=1):
        bundle = shutuba_scraper.fetch_shutuba_and_odds(
            race_id=r_item["race_id"],
            date_str=date_str,
        )
        df = bundle["shutuba_df"].copy()

        feat_df, _ = train_model.create_features(df, category_maps=category_maps)
        raw_probs = model.predict_proba(feat_df[feature_cols])[:, 1]
        raw_probs = np.clip(raw_probs, 1e-5, 1.0)
        ai_pure_prob = raw_probs / raw_probs.sum()

        win_odds = np.clip(pd.to_numeric(df["win_odds"], errors="coerce").fillna(25.0).to_numpy(dtype=float), 1.05, 999.9)
        mkt_raw = 0.80 / win_odds
        mkt_prob = mkt_raw / mkt_raw.sum()

        blended = (ai_weight * ai_pure_prob) + ((1.0 - ai_weight) * mkt_prob)
        blended = blended / blended.sum()

        df["win_odds"] = win_odds
        df["ai_p"] = blended
        df["mkt_q"] = mkt_prob
        df["edge_ratio"] = df["ai_p"] / np.maximum(df["mkt_q"], 1e-5)
        df["win_ev"] = df["ai_p"] * df["win_odds"]
        df["popularity"] = df["win_odds"].rank(method="min", ascending=True).astype(int)
        df = df.sort_values("ai_p", ascending=False).reset_index(drop=True)
        df["ai_rank"] = np.arange(1, len(df) + 1)

        entropy = compute_race_entropy(df["ai_p"].to_numpy(dtype=float))
        top1_p = float(df["ai_p"].iloc[0])
        top_gap = float(df["ai_p"].iloc[0] - (df["ai_p"].iloc[1] if len(df) > 1 else 0.0))

        first_row = df.iloc[0]
        r_title = str(first_row.get("race_title", r_item.get("race_title", "")) or "").strip()
        if not r_title or r_title == "nan":
            r_title = f"{r_item['venue']} {r_item['race_num']}R"

        evaluated.append({
            "leg": idx,
            "race_id": r_item["race_id"],
            "venue": str(first_row.get("venue", r_item["venue"])),
            "race_num": int(r_item["race_num"]),
            "race_title": r_title,
            "start_time": str(first_row.get("start_time", "") or ""),
            "surface": str(first_row.get("surface", "芝")),
            "distance": int(float(first_row.get("distance", 2000))),
            "track_condition": str(first_row.get("track_condition", "良")),
            "has_real_win": bool(bundle.get("has_real_win", False)),
            "entropy": entropy,
            "top1_p": top1_p,
            "top_gap": top_gap,
            "df": df,
        })

    # 発走時刻順に並べてWIN5の1〜5戦目を正確に振る
    if all(ev["start_time"] for ev in evaluated):
        evaluated.sort(key=lambda x: x["start_time"])
        for new_i, ev in enumerate(evaluated, start=1):
            ev["leg"] = new_i

    return evaluated


def _select_candidates_for_race(
    race_df: pd.DataFrame,
    style_mode: str,
    exclude_fav1: bool,
    force_ai1: bool = True,
    top_k: int = 6,
) -> pd.DataFrame:
    """
    レースごとのWIN5候補馬を順位付けする。
    ・超大穴のedge_ratio暴走（単勝180倍がAI1位を追い抜く現象）を防ぐため、
      実効エッジ比に上限キャップを設けつつ、AI勝率の重みを適正化する。
    ・force_ai1=True（初期設定）のときは、必ずAI勝率1位（1番人気除外時は実質AI最上位）を
      1枠目（1点突破・カバー筆頭）に固定し、2枠目以降に期待値×実力バランス上位を配置する。
    """
    df = race_df.copy()

    if exclude_fav1:
        pool = df[df["popularity"] != 1].copy()
        if len(pool) < top_k:
            pool = df.copy()
    else:
        pool = df.copy()

    if style_mode == "solid":
        edge_cap = 2.0
        min_p = 0.055
        p_weight = 1.60
    elif style_mode == "upset":
        edge_cap = 3.8
        min_p = 0.030
        p_weight = 0.85
    else:
        edge_cap = 2.8
        min_p = 0.040
        p_weight = 1.25

    eff_edge = np.minimum(pool["edge_ratio"].to_numpy(dtype=float), edge_cap)
    ai_p_arr = np.clip(pool["ai_p"].to_numpy(dtype=float), 1e-5, 1.0)
    pool["win5_score"] = (ai_p_arr ** p_weight) * (eff_edge ** 0.70)

    # AI1位（pool内の勝率最上位馬）を特定
    pool_by_ai = pool.sort_values("ai_p", ascending=False).reset_index(drop=True)
    ai1_row = pool_by_ai.iloc[[0]].copy()
    ai1_hnum = int(ai1_row["horse_num"].iloc[0])

    # 2番手以降の候補（極端な低確率馬を除外しつつ妙味＋実力で順位付け）
    others = pool[pool["horse_num"].astype(int) != ai1_hnum].copy()
    others_filtered = others[others["ai_p"] >= min_p]
    if len(others_filtered) >= max(top_k - 1, 2):
        others = others_filtered

    if force_ai1:
        others_sorted = others.sort_values("win5_score", ascending=False).head(max(top_k - 1, 1))
        combined = pd.concat([ai1_row, others_sorted], ignore_index=True)
        return combined.head(top_k).reset_index(drop=True)

    filtered_all = pool[pool["ai_p"] >= min_p]
    if len(filtered_all) >= top_k:
        pool = filtered_all
    return pool.sort_values("win5_score", ascending=False).head(top_k).reset_index(drop=True)


def build_win5_recommendations(
    evaluated_races: list[dict],
    style_mode: str = "balanced",
    exclude_fav1: bool = False,
    carryover_boost: float = 1.0,
    force_ai1: bool = True,
) -> dict:
    """
    3通り（300円）・5通り（500円）・10通り（1000円）について、
    「フォーメーション最適化」と「単品厳選（バラ買い）最適化」の両方を算出して返す。
    """
    payout_rate = 0.70 * carryover_boost
    if style_mode == "solid":
        prob_exp = 1.10
        ev_exp = 0.45
        min_ev_single = 0.75
    elif style_mode == "upset":
        prob_exp = 0.45
        ev_exp = 0.90
        min_ev_single = 1.10
    else:
        prob_exp = 0.78
        ev_exp = 0.65
        min_ev_single = 0.90

    cands_list_6 = [
        _select_candidates_for_race(
            ev["df"],
            style_mode=style_mode,
            exclude_fav1=exclude_fav1,
            force_ai1=force_ai1,
            top_k=6,
        )
        for ev in evaluated_races
    ]
    cands_list_5 = [c.head(5) for c in cands_list_6]

    def solve_formation(slot_pattern: tuple[int, int, int, int, int]) -> dict:
        unique_perms = sorted(set(itertools.permutations(slot_pattern)))
        best_obj = -1.0
        best_res = None

        for perm in unique_perms:
            picked_per_race = []
            valid = True
            for r_idx, k_need in enumerate(perm):
                cdf = cands_list_6[r_idx]
                if len(cdf) < k_need:
                    valid = False
                    break
                picked_per_race.append(cdf.head(k_need))
            if not valid:
                continue

            row_lists = [[row for _, row in pdf.iterrows()] for pdf in picked_per_race]
            combos = list(itertools.product(*row_lists))

            total_prob = 0.0
            ev_sum = 0.0
            capped_ev_sum = 0.0
            ticket_rows = []

            for comb in combos:
                p_prod = 1.0
                q_prod = 1.0
                fav1_cnt = 0
                nums = []
                names = []
                for h_row in comb:
                    p_prod *= float(h_row["ai_p"])
                    q_prod *= float(h_row["mkt_q"])
                    if int(h_row["popularity"]) == 1:
                        fav1_cnt += 1
                    nums.append(int(h_row["horse_num"]))
                    names.append(str(h_row["horse_name"]))

                est_odds = min(payout_rate / max(q_prod, 1e-10), 6_000_000.0)
                est_payout = int(round(est_odds * 100.0, -1))
                ev = p_prod * est_odds

                total_prob += p_prod
                ev_sum += ev
                capped_ev_sum += min(ev, 25.0)
                ticket_rows.append({
                    "組合せ": "-".join(f"{n:02d}" for n in nums),
                    "1着馬（1戦目〜5戦目）": " / ".join(f"{n:02d}{nm}" for n, nm in zip(nums, names)),
                    "AI的中率(%)": p_prod * 100.0,
                    "想定払戻(円)": est_payout,
                    "期待値(EV)": ev,
                    "購入額(円)": 100,
                    "_fav1_cnt": fav1_cnt,
                })

            avg_ev = ev_sum / max(len(combos), 1)
            avg_capped_ev = capped_ev_sum / max(len(combos), 1)
            obj_score = (max(total_prob, 1e-9) ** prob_exp) * (max(avg_capped_ev, 0.1) ** ev_exp)

            if style_mode == "upset":
                avg_fav1 = float(np.mean([t["_fav1_cnt"] for t in ticket_rows]))
                if avg_fav1 >= 3.5:
                    obj_score *= 0.70

            if obj_score > best_obj:
                best_obj = obj_score
                t_df = (
                    pd.DataFrame(ticket_rows)
                    .sort_values(["AI的中率(%)", "期待値(EV)"], ascending=[False, False])
                    .drop(columns=["_fav1_cnt"])
                    .reset_index(drop=True)
                )
                best_res = {
                    "allocation": perm,
                    "picked_per_race": picked_per_race,
                    "tickets_df": t_df,
                    "total_cost": len(t_df) * 100,
                    "total_prob_pct": total_prob * 100.0,
                    "avg_ev": avg_ev,
                    "max_payout": int(t_df["想定払戻(円)"].max()),
                    "min_payout": int(t_df["想定払戻(円)"].min()),
                }

        return best_res

    # 単品厳選用：5^5 = 最大3,125通りを全件計算
    row_lists_5 = [[row for _, row in cdf.iterrows()] for cdf in cands_list_5]
    ai1_nums = [int(cdf["horse_num"].iloc[0]) for cdf in cands_list_5]

    all_single_rows = []
    for comb in itertools.product(*row_lists_5):
        p_prod = 1.0
        q_prod = 1.0
        fav1_cnt = 0
        ai1_match_cnt = 0
        nums = []
        names = []
        for r_i, h_row in enumerate(comb):
            h_num = int(h_row["horse_num"])
            p_prod *= float(h_row["ai_p"])
            q_prod *= float(h_row["mkt_q"])
            if int(h_row["popularity"]) == 1:
                fav1_cnt += 1
            if h_num == ai1_nums[r_i]:
                ai1_match_cnt += 1
            nums.append(h_num)
            names.append(str(h_row["horse_name"]))

        est_odds = min(payout_rate / max(q_prod, 1e-10), 6_000_000.0)
        est_payout = int(round(est_odds * 100.0, -1))
        ev = p_prod * est_odds
        capped_ev = min(ev, 25.0)
        score = (max(p_prod, 1e-9) ** prob_exp) * (max(capped_ev, 0.1) ** ev_exp)
        if style_mode == "upset" and fav1_cnt >= 4:
            score *= 0.60

        all_single_rows.append({
            "組合せ": "-".join(f"{n:02d}" for n in nums),
            "1着馬（1戦目〜5戦目）": " / ".join(f"{n:02d}{nm}" for n, nm in zip(nums, names)),
            "AI的中率(%)": p_prod * 100.0,
            "想定払戻(円)": est_payout,
            "期待値(EV)": ev,
            "購入額(円)": 100,
            "_ai1_cnt": ai1_match_cnt,
            "_score": score,
        })

    all_single_df = pd.DataFrame(all_single_rows).sort_values("_score", ascending=False)
    ev_filtered_df = all_single_df[all_single_df["期待値(EV)"] >= min_ev_single]

    def solve_singles(n_tickets: int) -> dict:
        src = ev_filtered_df if len(ev_filtered_df) >= n_tickets else all_single_df
        if force_ai1:
            main_line = all_single_df[all_single_df["_ai1_cnt"] == 5].head(1)
            if not main_line.empty:
                main_combo = str(main_line["組合せ"].iloc[0])
                rest = src[src["組合せ"] != main_combo].head(max(n_tickets - 1, 0))
                picked = pd.concat([main_line, rest], ignore_index=True)
            else:
                picked = src.head(n_tickets).copy()
        else:
            picked = src.head(n_tickets).copy()

        picked = picked.drop(columns=["_score", "_ai1_cnt"]).reset_index(drop=True)
        return {
            "tickets_df": picked,
            "total_cost": len(picked) * 100,
            "total_prob_pct": float(picked["AI的中率(%)"].sum()),
            "avg_ev": float(picked["期待値(EV)"].mean()),
            "max_payout": int(picked["想定払戻(円)"].max()),
            "min_payout": int(picked["想定払戻(円)"].min()),
        }

    return {
        "formation": {
            "3": solve_formation((1, 1, 1, 1, 3)),
            "5": solve_formation((1, 1, 1, 1, 5)),
            "10": solve_formation((1, 1, 1, 2, 5)),
        },
        "singles": {
            "3": solve_singles(3),
            "5": solve_singles(5),
            "10": solve_singles(10),
        },
    }


def build_win5_ipat_memo(
    evaluated_races: list[dict],
    date_str: str,
    mode_key: str,
    buy_style: str,
    res: dict,
) -> str:
    style_txt = "フォーメーション最適化" if buy_style == "formation" else "単品厳選（バラ買い）"
    lines = [
        f"【KAZUYAN AI WIN5期待値予想】 {date_str} / {mode_key}通り（{res['total_cost']:,}円）",
        f"買い方: {style_txt}",
        f"合成的中率: {res['total_prob_pct']:.4f}% / 平均期待値(EV): {res['avg_ev']:.2f} / 想定払戻: {res['min_payout']:,}〜{res['max_payout']:,}円",
        "-" * 54,
    ]
    if buy_style == "formation":
        alloc_str = " x ".join(str(x) for x in res["allocation"])
        lines.append(f"■ フォーメーション構成: {alloc_str} = {mode_key}通り")
        for ev_r, pdf in zip(evaluated_races, res["picked_per_race"]):
            h_parts = []
            for _, r in pdf.iterrows():
                h_num = int(r["horse_num"])
                h_name = str(r["horse_name"])
                h_odds = float(r["win_odds"])
                h_rank = int(r["ai_rank"])
                h_parts.append(f"{h_num:02d} {h_name}[AI{h_rank}位]({h_odds:.1f}倍)")
            h_txt = ", ".join(h_parts)
            lines.append(f"  {ev_r['leg']}戦目 [{ev_r['venue']}{ev_r['race_num']}R {ev_r['race_title']}]: {h_txt}")
    else:
        lines.append("■ 単品厳選リスト（各100円）:")
        for idx, r in res["tickets_df"].iterrows():
            comb_s = str(r["組合せ"])
            horses_s = str(r["1着馬（1戦目〜5戦目）"])
            pay_i = int(r["想定払戻(円)"])
            ev_f = float(r["期待値(EV)"])
            lines.append(f"  {idx + 1:02d}. [{comb_s}] {horses_s} | 想定 {pay_i:,}円 (EV {ev_f:.2f})")
    return "\n".join(lines)


def render_win5_tab(
    model,
    feature_cols: list[str],
    category_maps: dict,
    train_model,
    shutuba_scraper,
    ai_weight: float,
    selected_style_mode: str,
    exclude_fav1: bool,
    style_short_badge: str,
):
    """03_app.py の WIN5タブ内を描画するメイン関数"""
    st.markdown("### WIN5 期待値最適化ツール（3通り・5通り・10通り）")
    st.caption(
        "対象5レースのAI勝率と単勝オッズから『WIN5想定配当・期待値(EV)・各レース混戦度』を算出し、"
        "堅いレースはAI1位で1点突破・混戦レースにAI1位＋妙味馬を厚く配分する最適買い目を自動生成します。"
    )

    c_date, c_co, c_btn = st.columns([1.4, 1.4, 1.8])
    with c_date:
        win5_date = st.date_input("WIN5 対象日（土日祝開催日）", value=get_default_win5_date(), key="win5_date_input")
        win5_date_str = win5_date.strftime("%Y-%m-%d")
    with c_co:
        carryover_opt = st.selectbox(
            "キャリーオーバー補正",
            ["通常（払戻率70%）", "キャリーオーバー小（実質85%）", "キャリーオーバー大（実質100%）"],
            index=0,
            key="win5_co_select",
        )
        co_boost = 1.0 if "70%" in carryover_opt else (85.0 / 70.0 if "85%" in carryover_opt else 100.0 / 70.0)
    with c_btn:
        st.markdown("<div style='height:27px;'></div>", unsafe_allow_html=True)
        fetch_clicked = st.button("WIN5対象5レースを一括取得＆AI計算", type="primary", width="stretch", key="btn_fetch_win5")

    if fetch_clicked:
        with st.spinner(f"{win5_date_str} のWIN5対象5レースを取得してAI期待値計算中..."):
            try:
                targets, all_races, src_msg = fetch_win5_target_races(win5_date_str, shutuba_scraper)
                if len(targets) < 5:
                    st.warning(f"対象5レースを取得できませんでした（{src_msg}）。土日祝のJRA開催日を選択してください。")
                else:
                    evaluated = evaluate_win5_races(
                        target_races=targets,
                        date_str=win5_date_str,
                        model=model,
                        feature_cols=feature_cols,
                        category_maps=category_maps,
                        train_model=train_model,
                        shutuba_scraper=shutuba_scraper,
                        ai_weight=ai_weight,
                    )
                    st.session_state["win5_evaluated"] = evaluated
                    st.session_state["win5_all_races"] = all_races
                    st.session_state["win5_loaded_date"] = win5_date_str
                    st.session_state["win5_src_msg"] = src_msg
                    st.rerun()
            except Exception as exc:
                st.error(f"WIN5データ取得エラー: {exc}")

    evaluated = st.session_state.get("win5_evaluated")
    if not evaluated or len(evaluated) != 5:
        st.info("上の「WIN5対象5レースを一括取得＆AI計算」ボタンを押すと、対象5レースのAI分析と3通り・5通り・10通りの最適買い目が生成されます。")
        return

    loaded_date = st.session_state.get("win5_loaded_date", win5_date_str)
    src_msg = st.session_state.get("win5_src_msg", "")
    all_races = st.session_state.get("win5_all_races", [])

    # サイドバーの AI独自度 を動かした際にリアルタイム再計算
    re_evaluated = []
    for ev in evaluated:
        df_c = ev["df"].copy()
        feat_df, _ = train_model.create_features(df_c, category_maps=category_maps)
        raw_p = np.clip(model.predict_proba(feat_df[feature_cols])[:, 1], 1e-5, 1.0)
        ai_pure = raw_p / raw_p.sum()
        mkt_raw = 0.80 / np.clip(df_c["win_odds"].to_numpy(dtype=float), 1.05, 999.9)
        mkt_p = mkt_raw / mkt_raw.sum()
        blended = (ai_weight * ai_pure) + ((1.0 - ai_weight) * mkt_p)
        blended = blended / blended.sum()
        df_c["ai_p"] = blended
        df_c["mkt_q"] = mkt_p
        df_c["edge_ratio"] = df_c["ai_p"] / np.maximum(df_c["mkt_q"], 1e-5)
        df_c["win_ev"] = df_c["ai_p"] * df_c["win_odds"]
        df_c = df_c.sort_values("ai_p", ascending=False).reset_index(drop=True)
        df_c["ai_rank"] = np.arange(1, len(df_c) + 1)

        ev_copy = dict(ev)
        ev_copy["df"] = df_c
        ev_copy["entropy"] = compute_race_entropy(df_c["ai_p"].to_numpy(dtype=float))
        ev_copy["top1_p"] = float(df_c["ai_p"].iloc[0])
        ev_copy["top_gap"] = float(df_c["ai_p"].iloc[0] - (df_c["ai_p"].iloc[1] if len(df_c) > 1 else 0.0))
        re_evaluated.append(ev_copy)

    # 対象5レースの手動差し替え（必要時のみ）
    if all_races and len(all_races) >= 5:
        with st.expander(f"対象5レースの確認・手動変更（現在: {src_msg}）"):
            label_map = {r["label"]: r for r in all_races}
            all_labels = list(label_map.keys())
            sel_cols = st.columns(5)
            chosen_items = []
            for i_leg, ev in enumerate(re_evaluated):
                cur_idx = next((idx for idx, r in enumerate(all_races) if r["race_id"] == ev["race_id"]), 0)
                with sel_cols[i_leg]:
                    lbl = st.selectbox(f"{i_leg + 1}戦目", all_labels, index=cur_idx, key=f"win5_override_{i_leg}")
                    chosen_items.append(label_map[lbl])
            if st.button("選択した5レースで再計算する", key="btn_win5_override_apply"):
                with st.spinner("指定した5レースを取得・再計算中..."):
                    new_eval = evaluate_win5_races(
                        target_races=chosen_items,
                        date_str=loaded_date,
                        model=model,
                        feature_cols=feature_cols,
                        category_maps=category_maps,
                        train_model=train_model,
                        shutuba_scraper=shutuba_scraper,
                        ai_weight=ai_weight,
                    )
                    st.session_state["win5_evaluated"] = new_eval
                    st.session_state["win5_src_msg"] = "手動指定5レース"
                    st.rerun()

    # 5レース混戦度診断カード
    st.markdown(f"#### 対象5レース AI混戦度診断（{loaded_date}）")
    leg_cols = st.columns(5)
    for col_box, ev in zip(leg_cols, re_evaluated):
        chaos_tag, chaos_color = classify_race_chaos(ev["top1_p"], ev["top_gap"], ev["entropy"])

        top_h = ev["df"].iloc[0]
        surf_s = "ダ" if "ダ" in ev["surface"] else "芝"
        odds_flag = "実オッズ" if ev["has_real_win"] else "仮オッズ"
        start_t_esc = html.escape(ev["start_time"])
        venue_esc = html.escape(ev["venue"])
        title_esc = html.escape(ev["race_title"][:11])
        top_num = int(top_h["horse_num"])
        top_name_esc = html.escape(str(top_h["horse_name"])[:8])
        top_pct = float(top_h["ai_p"]) * 100.0

        card_html = (
            f'<div style="background:#ffffff; border:1px solid #cbd5e1; border-top:4px solid {chaos_color}; '
            'border-radius:8px; padding:10px 12px; min-height:132px;">'
            f'<div style="font-size:0.78rem; color:#64748b; font-weight:700;">WIN5 {ev["leg"]}戦目 {start_t_esc}</div>'
            f'<div style="font-size:0.95rem; font-weight:800; color:#192442; margin:2px 0;">{venue_esc}{ev["race_num"]}R {title_esc}</div>'
            f'<div style="font-size:0.76rem; color:#475569;">{surf_s}{ev["distance"]}m ({odds_flag})</div>'
            f'<div style="font-size:0.76rem; font-weight:800; color:{chaos_color}; margin-top:4px;">{chaos_tag}</div>'
            f'<div style="font-size:0.78rem; color:#0f172a; margin-top:4px;">AI1位: <strong>{top_num:02d} {top_name_esc}</strong> ({top_pct:.1f}%)</div>'
            '</div>'
        )
        col_box.markdown(card_html, unsafe_allow_html=True)

    # 3通り / 5通り / 10通り 選択タイル
    st.markdown("#### WIN5 買い目点数モード選択")
    if "win5_mode_key" not in st.session_state:
        st.session_state["win5_mode_key"] = "5"

    cur_w5_mode = st.session_state["win5_mode_key"]
    st.markdown('<div class="mode-btn-zone"></div>', unsafe_allow_html=True)
    wb1, wb2, wb3 = st.columns(3)
    with wb1:
        if st.button("3通り (300円)", key="btn_w5_3", type="primary" if cur_w5_mode == "3" else "secondary", width="stretch"):
            st.session_state["win5_mode_key"] = "3"
            st.rerun()
    with wb2:
        if st.button("5通り (500円)", key="btn_w5_5", type="primary" if cur_w5_mode == "5" else "secondary", width="stretch"):
            st.session_state["win5_mode_key"] = "5"
            st.rerun()
    with wb3:
        if st.button("10通り (1,000円)", key="btn_w5_10", type="primary" if cur_w5_mode == "10" else "secondary", width="stretch"):
            st.session_state["win5_mode_key"] = "10"
            st.rerun()

    opt_c1, opt_c2 = st.columns([2.2, 1.3])
    with opt_c1:
        buy_style_label = st.radio(
            "買い目構築方式",
            [
                "フォーメーション最適化（即PAT一括入力向き：1×1×1×1×3 / 1×1×1×1×5 / 1×1×1×2×5）",
                "単品厳選（バラ買いEV特化：全3,125通りから期待値×的中率トップを直接抽出）",
            ],
            index=0,
            horizontal=True,
            key="win5_buy_style_radio",
        )
    with opt_c2:
        force_ai1 = st.checkbox(
            "各レースのAI1位を買い目に必ず含める",
            value=True,
            key="win5_force_ai1_chk",
            help="ONにすると1点突破・複数カバー問わず全レースでAI勝率1位を筆頭採用し、残りの枠で期待値の高い穴馬をカバーします。",
        )

    buy_style_key = "formation" if "フォーメーション" in buy_style_label else "singles"

    recs = build_win5_recommendations(
        evaluated_races=re_evaluated,
        style_mode=selected_style_mode,
        exclude_fav1=exclude_fav1,
        carryover_boost=co_boost,
        force_ai1=force_ai1,
    )

    active_res = recs[buy_style_key][cur_w5_mode]
    t_df = active_res["tickets_df"]
    style_esc = html.escape(style_short_badge)
    total_cost_val = active_res["total_cost"]

    if buy_style_key == "formation":
        alloc = active_res["allocation"]
        alloc_str = " × ".join(str(x) for x in alloc)
        leg_lines_html = []
        for ev_r, k_n, pdf in zip(re_evaluated, alloc, active_res["picked_per_race"]):
            chip_parts = []
            for _, r in pdf.iterrows():
                h_n = int(r["horse_num"])
                h_nm = html.escape(str(r["horse_name"]))
                h_od = float(r["win_odds"])
                h_er = float(r["edge_ratio"])
                h_rk = int(r["ai_rank"])
                is_axis_chip = (k_n == 1 or h_rk == 1)
                chip_cls = "horse-chip horse-chip-axis" if is_axis_chip else "horse-chip"
                rk_tag = "AI1位" if h_rk == 1 else f"AI{h_rk}位"
                chip_parts.append(f'<span class="{chip_cls}">{h_n:02d} {h_nm} [{rk_tag}] ({h_od:.1f}倍 / EV比{h_er:.2f})</span>')
            chips_str = "".join(chip_parts)
            tag_txt = "1点突破" if k_n == 1 else f"{k_n}頭カバー"
            v_esc = html.escape(ev_r["venue"])
            leg_lines_html.append(
                f'<div style="margin-bottom:5px;"><strong>{ev_r["leg"]}戦目 {v_esc}{ev_r["race_num"]}R（{tag_txt}）：</strong> {chips_str}</div>'
            )
        legs_body = "".join(leg_lines_html)
        card_html = (
            '<div class="bet-card">'
            f'<div class="bet-card-title">WIN5 {cur_w5_mode}通り フォーメーション最適化（{alloc_str} ＝ {total_cost_val:,}円 / 方針: {style_esc}）</div>'
            f'{legs_body}'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    else:
        card_html = (
            '<div class="bet-card">'
            f'<div class="bet-card-title">WIN5 {cur_w5_mode}通り 単品厳選（計 {len(t_df)}点 × 100円 ＝ {total_cost_val:,}円 / 方針: {style_esc}）</div>'
            f'<div style="font-size:0.86rem; color:#475569;">5レース各上位候補の直積（最大3,125通り）から、AI1位本線を押さえつつリスク調整期待値が最も高い{cur_w5_mode}通りを厳選しています。</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("合計購入金額", f"{total_cost_val:,} 円（{cur_w5_mode}通り）")
    m2.metric("合成AI的中率", f"{active_res['total_prob_pct']:.4f}%")
    m3.metric("平均期待値 (EV)", f"{active_res['avg_ev']:.2f}")
    m4.metric("想定払戻レンジ", f"{active_res['min_payout']:,} 〜 {active_res['max_payout']:,} 円")

    st.dataframe(
        t_df.style.format({
            "AI的中率(%)": "{:.4f}%",
            "想定払戻(円)": "{:,}",
            "期待値(EV)": "{:.2f}",
            "購入額(円)": "{:,}",
        }),
        width="stretch",
        hide_index=True,
    )

    memo_txt = build_win5_ipat_memo(re_evaluated, loaded_date, cur_w5_mode, buy_style_key, active_res)
    with st.expander("即PAT WIN5入力用まとめテキスト（右上のアイコンでコピー可能）"):
        st.code(memo_txt, language="text")

    with st.expander("対象5レース別 AI評価・オッズ妙味（EV比）ランキング詳細"):
        sub_tabs = st.tabs([f"{ev['leg']}戦目 {ev['venue']}{ev['race_num']}R" for ev in re_evaluated])
        for st_tab, ev in zip(sub_tabs, re_evaluated):
            with st_tab:
                d = ev["df"]
                disp = pd.DataFrame({
                    "AI順位": d["ai_rank"].astype(int),
                    "馬番": d["horse_num"].astype(int),
                    "馬名": d["horse_name"],
                    "騎手": d["jockey_id"],
                    "人気": d["popularity"].astype(int),
                    "単勝オッズ": d["win_odds"],
                    "AI勝率(%)": d["ai_p"] * 100.0,
                    "市場支持率(%)": d["mkt_q"] * 100.0,
                    "WIN5エッジ比(P/q)": d["edge_ratio"],
                    "単勝EV": d["win_ev"],
                })
                st.dataframe(
                    disp.style.format({
                        "単勝オッズ": "{:.1f}",
                        "AI勝率(%)": "{:.1f}%",
                        "市場支持率(%)": "{:.1f}%",
                        "WIN5エッジ比(P/q)": "{:.2f}",
                        "単勝EV": "{:.2f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )