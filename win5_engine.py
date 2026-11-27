import html
import itertools
import re
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

JRA_WIN5_PAYOUT_RATE = 0.70
JRA_WIN5_CAP_YEN = 600_000_000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/",
}


def _decode_html(resp: requests.Response) -> str:
    raw = resp.content
    for enc in ("utf-8", "euc_jp", "cp932"):
        try:
            return raw.decode(enc, errors="strict")
        except UnicodeDecodeError:
            continue
    return raw.decode("euc_jp", errors="replace")


def detect_win5_races(date_str: str, race_list: list[dict]) -> tuple[list[dict], str]:
    """
    指定日のWIN5対象5レースを特定する。
    1. netkeiba WIN5ページから公式の5レースIDを取得
    2. 取得できない場合（過去日・未公開など）はJRA開催ルール（各場10R/11Rの後ろ5レース）から自動推定
    """
    ymd = date_str.replace("-", "")
    id_to_item = {r["race_id"]: r for r in race_list}

    for url in (
        f"https://race.netkeiba.com/top/win5.html?kaisai_date={ymd}",
        "https://race.netkeiba.com/top/win5.html",
    ):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(_decode_html(resp), "lxml")
            found_ids = []
            for a in soup.select('a[href*="race_id="]'):
                m = re.search(r"race_id=(\d{12})", a.get("href", ""))
                if not m:
                    continue
                rid = m.group(1)
                if rid in id_to_item and rid not in found_ids:
                    found_ids.append(rid)
                if len(found_ids) == 5:
                    break
            if len(found_ids) == 5:
                return [id_to_item[rid] for rid in found_ids], "公式WIN5対象レースを自動検出"
        except Exception:
            continue

    if len(race_list) <= 5:
        return list(race_list), "開催レース一覧から自動選択"

    venues = list(dict.fromkeys(r["venue"] for r in race_list))
    venue_priority = {"東京": 3, "中山": 3, "京都": 2, "阪神": 2, "中京": 2}

    def sort_key(r: dict):
        return (int(r["race_num"]), venue_priority.get(r["venue"], 1), int(r["race_id"][4:6]))

    if len(venues) >= 3:
        r11 = sorted([r for r in race_list if r["race_num"] == 11], key=sort_key)
        r10 = sorted([r for r in race_list if r["race_num"] == 10], key=sort_key)
        picked = r10[-2:] + r11[-3:]
    elif len(venues) == 2:
        r11 = sorted([r for r in race_list if r["race_num"] == 11], key=sort_key)
        r10 = sorted([r for r in race_list if r["race_num"] == 10], key=sort_key)
        r09 = sorted([r for r in race_list if r["race_num"] == 9], key=sort_key)
        picked = r09[-1:] + r10[-2:] + r11[-2:]
    else:
        picked = sorted(race_list, key=sort_key)[-5:]

    if len(picked) < 5:
        picked = sorted(race_list, key=sort_key)[-5:]

    return picked[:5], "JRA開催構成（10R/11R中心）から自動推定（下メニューで変更可）"


def enrich_win5_leg(
    shutuba_df: pd.DataFrame,
    train_model,
    model,
    feature_cols: list[str],
    category_maps: dict,
    ai_weight: float = 0.75,
    style_mode: str = "balanced",
    exclude_fav1: bool = False,
    exclude_top2: bool = False,
) -> pd.DataFrame:
    """1レース分の出馬表にAI勝率・市場支持率・期待値エッジ・混戦度・WIN5適性スコアを付与する"""
    df = shutuba_df.copy()

    feat_df, _ = train_model.create_features(df, category_maps=category_maps)
    raw_probs = model.predict_proba(feat_df[feature_cols])[:, 1]
    raw_probs = np.clip(raw_probs, 1e-5, 1.0)
    ai_pure_prob = raw_probs / raw_probs.sum()

    win_odds = np.clip(pd.to_numeric(df["win_odds"], errors="coerce").fillna(30.0).to_numpy(dtype=float), 1.05, 500.0)
    raw_q = 1.0 / win_odds
    mkt_prob = raw_q / raw_q.sum()

    blended_prob = (ai_weight * ai_pure_prob) + ((1.0 - ai_weight) * mkt_prob)
    blended_prob = blended_prob / blended_prob.sum()

    df["win_odds"] = win_odds
    df["popularity"] = df["win_odds"].rank(method="min", ascending=True).astype(int)
    df["ai_pure_prob"] = ai_pure_prob
    df["win_prob"] = blended_prob
    df["mkt_prob"] = mkt_prob
    df["ev_edge"] = df["win_prob"] / np.clip(df["mkt_prob"], 1e-5, 1.0)
    df["win_ev"] = df["win_prob"] * df["win_odds"]

    # 正規化エントロピー（0に近いほど1強、1に近いほど大混戦）
    n_h = max(2, len(df))
    chaos = float(-np.sum(blended_prob * np.log(np.clip(blended_prob, 1e-9, 1.0))) / np.log(n_h))
    df["race_chaos"] = round(chaos, 3)

    if style_mode == "solid":
        df["leg_score"] = (df["win_prob"] ** 1.05) * (np.clip(df["ev_edge"], 0.4, 3.0) ** 0.40)
    elif style_mode == "upset":
        df["leg_score"] = (df["win_prob"] ** 0.55) * (np.clip(df["ev_edge"], 0.4, 5.0) ** 1.25)
    else:
        df["leg_score"] = (df["win_prob"] ** 0.78) * (np.clip(df["ev_edge"], 0.4, 4.0) ** 0.85)

    if exclude_top2:
        mask_ex = df["popularity"].isin([1, 2])
        if int((~mask_ex).sum()) >= 5:
            df.loc[mask_ex, "leg_score"] *= 0.005
    elif exclude_fav1:
        mask_ex = df["popularity"] == 1
        if int((~mask_ex).sum()) >= 5:
            df.loc[mask_ex, "leg_score"] *= 0.005

    df = df.sort_values("leg_score", ascending=False).reset_index(drop=True)
    df["win5_rank"] = np.arange(1, len(df) + 1)
    return df


def _eval_single_combo(horses_tuple: tuple[dict, ...]) -> dict:
    p_combo = 1.0
    q_combo = 1.0
    for h in horses_tuple:
        p_combo *= float(h["win_prob"])
        q_combo *= float(h["mkt_prob"])

    est_payout = min(JRA_WIN5_CAP_YEN, max(100, int(round(100.0 * JRA_WIN5_PAYOUT_RATE / max(q_combo, 1e-10)))))
    est_odds = est_payout / 100.0
    ev = p_combo * est_odds
    edge_prod = p_combo / max(q_combo, 1e-10)
    return {
        "horses": horses_tuple,
        "hit_prob": p_combo,
        "mkt_prob": q_combo,
        "est_odds": est_odds,
        "est_payout": est_payout,
        "ev": ev,
        "edge_prod": edge_prod,
    }


def build_win5_plans(
    enriched_legs: list[pd.DataFrame],
    style_mode: str = "balanced",
    buy_method: str = "direct",
) -> dict:
    """
    3通り（300円）・5通り（500円）・10通り（1,000円）のWIN5買い目プランを生成する。
    buy_method:
      - "direct": 各レース上位6頭（最大7,776通り）から期待値×現実的中率スコア上位N点を個別厳選
      - "formation": N=3:(1,1,1,1,3), N=5:(1,1,1,1,5), N=10:(1,1,1,2,5) の全割当パターンを全探索最適化
    """
    if len(enriched_legs) != 5:
        raise ValueError("WIN5の計算には5レース分のデータが必要です。")

    plans = {}
    gamma_direct = 0.45 if style_mode == "solid" else (0.14 if style_mode == "upset" else 0.28)
    beta_form = 0.62 if style_mode == "solid" else (0.26 if style_mode == "upset" else 0.42)

    pool_direct = None
    if buy_method == "direct":
        pools = [leg.head(min(6, len(leg))).to_dict("records") for leg in enriched_legs]
        all_c = []
        for tup in itertools.product(*pools):
            c = _eval_single_combo(tup)
            c["rank_score"] = c["ev"] * (max(c["hit_prob"], 1e-9) ** gamma_direct)
            all_c.append(c)
        all_c.sort(key=lambda x: x["rank_score"], reverse=True)
        pool_direct = all_c

    alloc_map = {
        3: sorted(set(itertools.permutations([1, 1, 1, 1, 3]))),
        5: sorted(set(itertools.permutations([1, 1, 1, 1, 5]))),
        10: sorted(set(itertools.permutations([1, 1, 1, 2, 5]))),
    }

    for n_pts in (3, 5, 10):
        if buy_method == "direct":
            picked = pool_direct[:n_pts]
            total_hit_prob = float(sum(c["hit_prob"] for c in picked))
            mean_ev = float(np.mean([c["ev"] for c in picked])) if picked else 0.0
            exp_return = float(sum(c["ev"] for c in picked) * 100.0)

            used_per_leg = []
            for idx_l in range(5):
                nums = list(dict.fromkeys(int(c["horses"][idx_l]["horse_num"]) for c in picked))
                used_per_leg.append(nums)

            plans[str(n_pts)] = {
                "points": n_pts,
                "cost": n_pts * 100,
                "method": "direct",
                "alloc": tuple(len(x) for x in used_per_leg),
                "used_nums": used_per_leg,
                "combos": picked,
                "total_hit_prob": total_hit_prob,
                "mean_ev": mean_ev,
                "expected_return": exp_return,
            }
        else:
            best_obj = -1.0
            best_pack = None
            for alloc in alloc_map[n_pts]:
                if any(len(enriched_legs[i]) < alloc[i] for i in range(5)):
                    continue
                sel_legs = [enriched_legs[i].head(alloc[i]) for i in range(5)]
                hit_prob = 1.0
                for sl in sel_legs:
                    hit_prob *= float(sl["win_prob"].sum())

                combos = []
                ev_sum = 0.0
                for tup in itertools.product(*[sl.to_dict("records") for sl in sel_legs]):
                    c = _eval_single_combo(tup)
                    ev_sum += c["ev"]
                    combos.append(c)

                combos.sort(key=lambda x: x["ev"], reverse=True)
                mean_ev = ev_sum / len(combos)
                obj = (max(hit_prob, 1e-9) ** beta_form) * (max(mean_ev, 1e-6) ** (1.0 - beta_form * 0.45))

                if obj > best_obj:
                    best_obj = obj
                    best_pack = {
                        "points": n_pts,
                        "cost": n_pts * 100,
                        "method": "formation",
                        "alloc": alloc,
                        "used_nums": [sl["horse_num"].astype(int).tolist() for sl in sel_legs],
                        "combos": combos,
                        "total_hit_prob": hit_prob,
                        "mean_ev": mean_ev,
                        "expected_return": ev_sum * 100.0,
                    }
            plans[str(n_pts)] = best_pack

    return plans


def format_plan_dataframe(plan: dict, leg_labels: list[str]) -> pd.DataFrame:
    rows = []
    for idx, c in enumerate(plan["combos"], start=1):
        row = {"No": idx}
        for i, h in enumerate(c["horses"]):
            h_num = int(h["horse_num"])
            h_name = str(h["horse_name"])
            pop = int(h["popularity"])
            row[f"WIN{i + 1} ({leg_labels[i]})"] = f"{h_num:02d} {h_name} ({pop}人気)"
        row["AI的中率(%)"] = round(c["hit_prob"] * 100.0, 4)
        row["推定払戻(100円)"] = int(c["est_payout"])
        row["期待値(EV)"] = round(c["ev"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def build_win5_memo_text(plan: dict, leg_meta: list[dict]) -> str:
    header = " / ".join([f"WIN{i + 1}:{m['venue']}{m['race_num']}R" for i, m in enumerate(leg_meta)])
    method_label = "個別コンボ厳選" if plan["method"] == "direct" else f"フォーメーション({'x'.join(map(str, plan['alloc']))})"
    lines = [
        f"【JRA KEIBA AI WIN5予想】 {plan['points']}通り（{plan['cost']:,}円） / {method_label}",
        header,
        f"合計AI的中率: {plan['total_hit_prob'] * 100:.3f}% / 平均期待値(EV): {plan['mean_ev']:.2f}",
        "-" * 60,
    ]
    if plan["method"] == "formation":
        lines.append("■ フォーメーション入力用（馬番）")
        for i, nums in enumerate(plan["used_nums"]):
            n_str = ", ".join(f"{n:02d}" for n in sorted(nums))
            lines.append(f"  WIN{i + 1} ({leg_meta[i]['venue']}{leg_meta[i]['race_num']}R): {n_str}")
        lines.append("-" * 60)

    lines.append("■ 全買い目一覧（1点100円）")
    for idx, c in enumerate(plan["combos"], start=1):
        code = " - ".join(f"{int(h['horse_num']):02d}" for h in c["horses"])
        lines.append(
            f"  #{idx:02d}: [{code}]  想定払戻 {c['est_payout']:,}円 / EV {c['ev']:.2f} / 的中率 {c['hit_prob'] * 100:.4f}%"
        )
    return "\n".join(lines)


def render_win5_tab(
    shutuba_scraper,
    train_model,
    model,
    feature_cols: list[str],
    category_maps: dict,
    ai_weight: float,
    style_mode: str,
    exclude_fav1: bool,
    target_date_str: str,
):
    """03_app.py の WIN5タブ内を描画する"""
    st.markdown("### WIN5 AI期待値シミュレーター（3通り・5通り・10通り）")
    st.caption(
        "5レースのAI予測勝率と市場支持率（単勝オッズ逆算）の乖離から、各組合せのWIN5推定払戻・期待値(EV)を算出し、3通り・5通り・10通りの最適買い目を生成します。"
    )

    fetched_list = st.session_state.get("fetched_race_list", [])
    fetched_date = st.session_state.get("fetched_race_date", target_date_str)

    if not fetched_list:
        c_init1, c_init2 = st.columns([2, 3])
        with c_init1:
            if st.button("開催日のWIN5対象レースを取得", key="win5_init_fetch", type="primary", width="stretch"):
                with st.spinner(f"{target_date_str} の開催レースを取得中..."):
                    try:
                        races_found = shutuba_scraper.fetch_race_list_by_date(target_date_str)
                        st.session_state["fetched_race_list"] = races_found
                        st.session_state["fetched_race_date"] = target_date_str
                        if not races_found:
                            st.warning("指定日のJRA開催レースが見つかりません。左サイドバーから日曜（WIN5開催日）を選択してください。")
                        else:
                            st.rerun()
                    except Exception as e:
                        st.error(f"取得エラー: {e}")
        with c_init2:
            st.info("先に左サイドバーで日曜（WIN5開催日）を選んで「開催レース一覧を取得」するか、左のボタンを押してください。")
        return

    detect_key = f"{fetched_date}_{len(fetched_list)}"
    if st.session_state.get("win5_detect_key") != detect_key:
        auto_5, detect_msg = detect_win5_races(fetched_date, fetched_list)
        st.session_state["win5_target_races"] = auto_5
        st.session_state["win5_detect_msg"] = detect_msg
        st.session_state["win5_detect_key"] = detect_key
        st.session_state.pop("win5_raw_bundles", None)

    target_races = st.session_state.get("win5_target_races", [])
    detect_msg = st.session_state.get("win5_detect_msg", "")

    st.info(f"対象日: **{fetched_date}** ／ {detect_msg}")

    with st.expander("WIN5対象5レースの確認・変更（自動検出がズレた場合はここで調整）", expanded=False):
        label_map = {r["label"]: r for r in fetched_list}
        all_labels = list(label_map.keys())
        new_selected = []
        sel_cols = st.columns(5)
        for i in range(5):
            cur_r = target_races[i] if i < len(target_races) else fetched_list[-1]
            cur_idx = all_labels.index(cur_r["label"]) if cur_r["label"] in all_labels else 0
            with sel_cols[i]:
                picked_l = st.selectbox(f"WIN{i + 1}", all_labels, index=cur_idx, key=f"win5_leg_sel_{i}")
                new_selected.append(label_map[picked_l])
        if st.button("この5レースで確定する", key="win5_apply_custom_5"):
            st.session_state["win5_target_races"] = new_selected
            st.session_state.pop("win5_raw_bundles", None)
            st.rerun()

    opt_c1, opt_c2, opt_c3 = st.columns([2.2, 1.8, 2.0])
    with opt_c1:
        buy_method_label = st.radio(
            "買い目の組み立て方式",
            [
                "個別コンボ厳選（期待値最大化）",
                "フォーメーション最適化（即PAT一括入力向け）",
            ],
            index=0,
            key="win5_buy_method_radio",
        )
        buy_method = "direct" if "個別" in buy_method_label else "formation"
    with opt_c2:
        exclude_top2 = st.checkbox(
            "1・2番人気を除外（高配当特化）",
            value=False,
            key="win5_exclude_top2",
            help="全5レースで単勝1・2番人気の馬を買い目候補から外し、中穴〜大穴決着の高期待値ゾーンを狙います。",
        )
    with opt_c3:
        st.write("")
        run_fetch = st.button(
            "WIN5 5レース一括取得＆AI期待値計算",
            key="win5_run_all_btn",
            type="primary",
            width="stretch",
        )

    if run_fetch:
        if len(target_races) != 5:
            st.error("WIN5対象レースが5レース揃っていません。")
            return
        bundles = []
        prog = st.progress(0.0, text="WIN5対象5レースの出馬表とオッズを取得中...")
        try:
            for idx_r, r_item in enumerate(target_races):
                prog.progress((idx_r) / 5.0, text=f"WIN{idx_r + 1}: {r_item['venue']}{r_item['race_num']}R を取得中...")
                b = shutuba_scraper.fetch_shutuba_and_odds(r_item["race_id"], fetched_date)