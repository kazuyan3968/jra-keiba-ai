import importlib.util
import itertools
import os
import re
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="JRA KEIBA AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

JST = timezone(timedelta(hours=9))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "keiba_lgbm.pkl")
UPCOMING_CSV = os.path.join(BASE_DIR, "data", "upcoming_race.csv")
HIST_CSV = os.path.join(BASE_DIR, "data", "historical_races.csv")
BETS_CSV = os.path.join(BASE_DIR, "data", "bet_history.csv")


def _load_local_module(module_name: str, filename: str):
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_mod = _load_local_module("train_model", "02_train_model.py")
shutuba_scraper = _load_local_module("shutuba_scraper", "06_shutuba_scraper.py")
bet_tracker = _load_local_module("bet_tracker", "07_bet_tracker.py")
harville_mod = _load_local_module("harville_calculator", "harville_calculator.py")


# ---------------------------------------------------------
# スタイル定義（クリーン・テキスト重視・絵文字なし）
# ---------------------------------------------------------
st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
        max-width: 1220px;
    }
    .top-navbar {
        background: linear-gradient(90deg, #0d192e 0%, #162642 100%);
        color: #ffffff;
        padding: 14px 22px;
        border-radius: 10px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-bottom: 14px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.14);
    }
    .top-navbar-title {
        font-size: 1.32rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #ffffff;
    }
    .nav-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    .badge-green {
        background-color: #15803d;
        color: #ffffff;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-warn {
        background-color: #b45309;
        color: #ffffff;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .badge-slate {
        background-color: #293852;
        color: #e2e8f0;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #3c4f70;
    }
    .race-banner {
        background: linear-gradient(135deg, #2b1035 0%, #441950 100%);
        color: #ffffff;
        padding: 20px 26px;
        border-radius: 12px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        margin-top: 8px;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(43, 16, 53, 0.22);
    }
    .race-banner-sub {
        font-size: 1.02rem;
        font-weight: 700;
        color: #f3e8ff;
        margin-bottom: 4px;
    }
    .race-banner-time {
        font-size: 0.86rem;
        font-weight: 500;
        color: #d8b4fe;
        margin-left: 8px;
    }
    .race-banner-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.25;
        margin-bottom: 10px;
    }
    .race-banner-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        font-size: 0.88rem;
        color: #f3e8ff;
    }
    .cond-pill {
        background-color: #fde68a;
        color: #78350f;
        font-weight: 800;
        padding: 2px 10px;
        border-radius: 5px;
        font-size: 0.82rem;
    }
    .status-pill {
        background-color: rgba(236, 253, 245, 0.95);
        color: #065f46;
        border: 1px solid #34d399;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 5px;
        font-size: 0.82rem;
    }
    .gauge-group {
        display: flex;
        gap: 14px;
        align-items: center;
    }
    .ring-box {
        width: 74px;
        height: 74px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(15, 10, 25, 0.38);
    }
    .ring-orange {
        border: 3.5px solid #f59e0b;
    }
    .ring-green {
        border: 3.5px solid #10b981;
    }
    .ring-val-orange {
        font-size: 1.12rem;
        font-weight: 800;
        color: #fbbf24;
        line-height: 1.1;
    }
    .ring-val-green {
        font-size: 1.12rem;
        font-weight: 800;
        color: #34d399;
        line-height: 1.1;
    }
    .ring-label {
        font-size: 0.68rem;
        color: #e9d5ff;
        margin-top: 2px;
        font-weight: 600;
    }
    .section-heading {
        font-size: 1.08rem;
        font-weight: 800;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# モデル読み込み＆学習状態判定
# ---------------------------------------------------------
@st.cache_resource
def load_artifact() -> dict | None:
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=30)
def get_training_data_status(_artifact: dict | None = None) -> dict:
    if os.path.exists(HIST_CSV):
        try:
            df_h = pd.read_csv(HIST_CSV, usecols=["race_id", "date", "horse_name"])
            sample_ratio = df_h["horse_name"].astype(str).str.startswith("サンプルホース").mean()
            n_races = int(df_h["race_id"].nunique())
            min_d = str(df_h["date"].min())
            max_d = str(df_h["date"].max())
            if sample_ratio > 0.5:
                return {
                    "is_real": False,
                    "badge": "学習元: テスト用サンプルデータ",
                    "detail": f"仮想データ（{n_races:,}レース / {min_d} - {max_d}）で学習しています。",
                }
            return {
                "is_real": True,
                "badge": f"学習元: JRA実データ ({n_races:,}R)",
                "detail": f"実際のJRA過去レース結果（{n_races:,}レース / 期間: {min_d} - {max_d}）を使用して学習しています。",
            }
        except Exception:
            pass

    if _artifact and isinstance(_artifact, dict):
        jockeys = [str(j) for j in _artifact.get("category_maps", {}).get("jockey_id", [])]
        real_jockeys = [
            j for j in jockeys
            if not j.startswith("JOCKEY_") and j not in ("UNKNOWN_J", "UNKNOWN", "nan", "")
        ]
        if len(real_jockeys) >= 20:
            n_r = _artifact.get("n_races")
            badge_txt = f"学習元: JRA実データ ({int(n_r):,}R)" if n_r else "学習元: JRA実データモデル適用中"
            return {
                "is_real": True,
                "badge": badge_txt,
                "detail": "JRA実レース結果で学習済みのモデル（models/keiba_lgbm.pkl）を読み込んで予測しています。",
            }

    return {
        "is_real": False,
        "badge": "学習データ未確認",
        "detail": "models/keiba_lgbm.pkl を確認してください。",
    }


artifact = load_artifact()
if artifact is None:
    st.error("学習済みモデル（models/keiba_lgbm.pkl）が見つかりません。先に 02_train_model.py を実行してください。")
    st.stop()

model = artifact["model"]
feature_cols = artifact["feature_cols"]
category_maps = artifact["category_maps"]
train_status = get_training_data_status(artifact)


# ---------------------------------------------------------
# 特徴量生成フォールバック（02_train_model.py と完全互換）
# ---------------------------------------------------------
def _fallback_create_features(df: pd.DataFrame, cmaps: dict) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    num_defaults = {
        "prev1_rank": 7.0,
        "prev2_rank": 7.0,
        "prev3_rank": 7.0,
        "prev1_time_diff": 0.8,
        "distance": 1800.0,
        "frame_num": 4.0,
        "horse_num": 8.0,
        "weight_carried": 56.0,
        "horse_weight": 476.0,
        "weight_diff": 0.0,
    }
    for col, default_val in num_defaults.items():
        if col not in out.columns:
            out[col] = default_val
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default_val)

    out["avg_prev_rank"] = (out["prev1_rank"] + out["prev2_rank"] + out["prev3_rank"]) / 3.0
    out["best_prev_rank"] = out[["prev1_rank", "prev2_rank", "prev3_rank"]].min(axis=1)
    out["rank_trend"] = out["prev3_rank"] - out["prev1_rank"]

    if "race_id" in out.columns:
        race_avg_rank = out.groupby("race_id")["avg_prev_rank"].transform("mean")
        race_avg_tdiff = out.groupby("race_id")["prev1_time_diff"].transform("mean")
    else:
        race_avg_rank = out["avg_prev_rank"].mean()
        race_avg_tdiff = out["prev1_time_diff"].mean()

    out["rel_prev_rank"] = out["avg_prev_rank"] - race_avg_rank
    out["rel_time_diff"] = out["prev1_time_diff"] - race_avg_tdiff

    safe_hw = np.clip(out["horse_weight"].to_numpy(dtype=float), 350.0, 650.0)
    out["abs_weight_diff"] = out["weight_diff"].abs()
    out["weight_ratio"] = out["weight_carried"] / safe_hw

    cat_cols = ["venue", "surface", "track_condition", "jockey_id", "trainer_id"]
    for col in cat_cols:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        s = out[col].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
        top_cats = cmaps.get(col, ["OTHER"])
        s_mapped = s.where(s.isin(top_cats), "OTHER")
        out[col] = pd.Categorical(s_mapped, categories=top_cats)

    return out, cmaps


# ---------------------------------------------------------
# ハービル確率・期待値・買い目生成エンジン
# ---------------------------------------------------------
def compute_harville_probabilities(win_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ハービル公式により各馬の連対率(2着以内)・複勝率(3着以内)を厳密計算する"""
    n = len(win_probs)
    p = np.clip(np.asarray(win_probs, dtype=float), 1e-6, 1.0)
    p = p / p.sum()

    quinella_prob = p.copy()
    place_prob = p.copy()

    if n < 2:
        return quinella_prob, place_prob

    for j in range(n):
        p2_j = 0.0
        for i in range(n):
            if i == j:
                continue
            denom1 = max(1e-6, 1.0 - p[i])
            p2_j += p[i] * (p[j] / denom1)
        quinella_prob[j] = p[j] + p2_j

    if n < 3:
        return quinella_prob, quinella_prob.copy()

    for k in range(n):
        p3_k = 0.0
        for i in range(n):
            if i == k:
                continue
            denom1 = max(1e-6, 1.0 - p[i])
            for j in range(n):
                if j == i or j == k:
                    continue
                denom2 = max(1e-6, 1.0 - p[i] - p[j])
                p3_k += p[i] * (p[j] / denom1) * (p[k] / denom2)
        place_prob[k] = quinella_prob[k] + p3_k

    return np.clip(quinella_prob, 0.0, 0.999), np.clip(place_prob, 0.0, 0.999)


def generate_ev_tickets(
    pred_df: pd.DataFrame,
    pair_odds_df: pd.DataFrame | None,
    exacta_odds_df: pd.DataFrame | None,
    trio_odds_df: pd.DataFrame | None,
    strategy: str,
    budget_yen: int,
    min_ev: float,
) -> pd.DataFrame:
    """ハービル確率と実オッズ（未取得時は合成オッズ）から全券種の期待値計算・資金配分を行う"""
    df = pred_df.sort_values("horse_num").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    nums = df["horse_num"].astype(int).tolist()
    names = dict(zip(nums, df["horse_name"].astype(str)))
    win_odds_map = dict(zip(nums, df["win_odds"].astype(float)))
    p_map = dict(zip(nums, df["ai_win_prob"].astype(float)))
    pop_rank_map = dict(zip(nums, df["popularity"].astype(int)))

    excluded_axis = set()
    if "穴狙い" in strategy:
        excluded_axis = {h for h, r in pop_rank_map.items() if r in (1, 2)}

    umaren_real = {}
    wide_real = {}
    if pair_odds_df is not None and not pair_odds_df.empty:
        for _, r in pair_odds_df.iterrows():
            k = (int(min(r["horse_num_1"], r["horse_num_2"])), int(max(r["horse_num_1"], r["horse_num_2"])))
            if "umaren_odds" in r and pd.notna(r["umaren_odds"]):
                umaren_real[k] = float(r["umaren_odds"])
            if "wide_odds" in r and pd.notna(r["wide_odds"]):
                wide_real[k] = float(r["wide_odds"])

    exacta_real = {}
    if exacta_odds_df is not None and not exacta_odds_df.empty:
        for _, r in exacta_odds_df.iterrows():
            k = (int(r["horse_num_1"]), int(r["horse_num_2"]))
            if "exacta_odds" in r and pd.notna(r["exacta_odds"]):
                exacta_real[k] = float(r["exacta_odds"])

    trio_real = {}
    if trio_odds_df is not None and not trio_odds_df.empty:
        for _, r in trio_odds_df.iterrows():
            k = tuple(sorted([int(r["horse_num_1"]), int(r["horse_num_2"]), int(r["horse_num_3"])]))
            if "trio_odds" in r and pd.notna(r["trio_odds"]):
                trio_real[k] = float(r["trio_odds"])

    raw_mkt = np.array([1.0 / max(1.05, win_odds_map[h]) for h in nums], dtype=float)
    q_mkt = dict(zip(nums, raw_mkt / raw_mkt.sum()))

    candidates = []

    # 1. 単勝
    for h in nums:
        if h in excluded_axis:
            continue
        prob = p_map[h]
        odds = win_odds_map[h]
        candidates.append({
            "bet_type": "単勝",
            "combo": f"{h:02d}",
            "horses": names[h],
            "prob": prob,
            "odds": odds,
            "ev": prob * odds,
            "is_real_odds": True,
        })

    top_horses = df.sort_values("ai_win_prob", ascending=False)["horse_num"].astype(int).head(8).tolist()

    # 2. 馬連・ワイド・馬単
    for h1, h2 in itertools.combinations(sorted(top_horses), 2):
        p1, p2 = p_map[h1], p_map[h2]
        p_12 = p1 * (p2 / max(1e-6, 1.0 - p1))
        p_21 = p2 * (p1 / max(1e-6, 1.0 - p2))
        prob_umaren = p_12 + p_21

        if (h1, h2) in umaren_real:
            o_umaren = umaren_real[(h1, h2)]
            real_u = True
        else:
            qm1, qm2 = q_mkt[h1], q_mkt[h2]
            qm_u = qm1 * (qm2 / max(1e-6, 1.0 - qm1)) + qm2 * (qm1 / max(1e-6, 1.0 - qm2))
            o_umaren = round(float(np.clip(0.775 / max(1e-5, qm_u), 1.5, 999.9)), 1)
            real_u = False

        if not ("穴狙い" in strategy and (h1 in excluded_axis or h2 in excluded_axis)):
            candidates.append({
                "bet_type": "馬連",
                "combo": f"{h1:02d}-{h2:02d}",
                "horses": f"{names[h1]} - {names[h2]}",
                "prob": prob_umaren,
                "odds": o_umaren,
                "ev": prob_umaren * o_umaren,
                "is_real_odds": real_u,
            })

        prob_wide = prob_umaren
        for h3 in nums:
            if h3 == h1 or h3 == h2:
                continue
            p3 = p_map[h3]
            d1 = max(1e-6, 1.0 - p3)
            prob_wide += p3 * (p1 / d1) * (p2 / max(1e-6, d1 - p1))
            prob_wide += p3 * (p2 / d1) * (p1 / max(1e-6, d1 - p2))
            d_1 = max(1e-6, 1.0 - p1)
            prob_wide += p1 * (p3 / d_1) * (p2 / max(1e-6, d_1 - p3))
            d_2 = max(1e-6, 1.0 - p2)
            prob_wide += p2 * (p3 / d_2) * (p1 / max(1e-6, d_2 - p3))

        if (h1, h2) in wide_real:
            o_wide = wide_real[(h1, h2)]
            real_w = True
        else:
            o_wide = round(float(np.clip(o_umaren * 0.36, 1.1, 250.0)), 1)
            real_w = False

        if not ("穴狙い" in strategy and (h1 in excluded_axis or h2 in excluded_axis)):
            candidates.append({
                "bet_type": "ワイド",
                "combo": f"{h1:02d}-{h2:02d}",
                "horses": f"{names[h1]} - {names[h2]}",
                "prob": prob_wide,
                "odds": o_wide,
                "ev": prob_wide * o_wide,
                "is_real_odds": real_w,
            })

        for first_h, second_h, p_ex in ((h1, h2, p_12), (h2, h1, p_21)):
            if first_h in excluded_axis:
                continue
            if (first_h, second_h) in exacta_real:
                o_ex = exacta_real[(first_h, second_h)]
                real_e = True
            else:
                qm1, qm2 = q_mkt[first_h], q_mkt[second_h]
                qm_e = qm1 * (qm2 / max(1e-6, 1.0 - qm1))
                o_ex = round(float(np.clip(0.75 / max(1e-5, qm_e), 2.0, 1999.9)), 1)
                real_e = False

            candidates.append({
                "bet_type": "馬単",
                "combo": f"{first_h:02d}->{second_h:02d}",
                "horses": f"{names[first_h]} -> {names[second_h]}",
                "prob": p_ex,
                "odds": o_ex,
                "ev": p_ex * o_ex,
                "is_real_odds": real_e,
            })

    # 3. 三連複
    top_trio_horses = sorted(top_horses[:7])
    for h1, h2, h3 in itertools.combinations(top_trio_horses, 3):
        if "穴狙い" in strategy and (h1 in excluded_axis or h2 in excluded_axis or h3 in excluded_axis):
            continue
        prob_trio = 0.0
        for a, b, c in itertools.permutations((h1, h2, h3)):
            pa, pb, pc = p_map[a], p_map[b], p_map[c]
            d1 = max(1e-6, 1.0 - pa)
            d2 = max(1e-6, 1.0 - pa - pb)
            prob_trio += pa * (pb / d1) * (pc / d2)

        key3 = (h1, h2, h3)
        if key3 in trio_real:
            o_trio = trio_real[key3]
            real_t = True
        else:
            qm_trio = 0.0
            for a, b, c in itertools.permutations((h1, h2, h3)):
                qa, qb, qc = q_mkt[a], q_mkt[b], q_mkt[c]
                qm_trio += qa * (qb / max(1e-6, 1.0 - qa)) * (qc / max(1e-6, 1.0 - qa - qb))
            o_trio = round(float(np.clip(0.75 / max(1e-6, qm_trio), 3.0, 4999.9)), 1)
            real_t = False

        candidates.append({
            "bet_type": "三連複",
            "combo": f"{h1:02d}-{h2:02d}-{h3:02d}",
            "horses": f"{names[h1]} - {names[h2]} - {names[h3]}",
            "prob": prob_trio,
            "odds": o_trio,
            "ev": prob_trio * o_trio,
            "is_real_odds": real_t,
        })

    if not candidates:
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates)

    if "堅実" in strategy:
        cand_df["rank_score"] = cand_df["ev"] * np.sqrt(np.clip(cand_df["prob"], 0.005, 1.0))
    else:
        cand_df["rank_score"] = cand_df["ev"] * (np.clip(cand_df["prob"], 0.005, 1.0) ** 0.25)

    cand_df = cand_df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    n_units = max(1, budget_yen // 100)
    if budget_yen <= 300:
        max_tickets = min(3, n_units)
    elif budget_yen <= 500:
        max_tickets = min(5, n_units)
    else:
        max_tickets = min(10, n_units)

    picked = cand_df[cand_df["ev"] >= min_ev].head(max_tickets).copy()
    if picked.empty:
        picked = cand_df.head(min(3, max_tickets)).copy()

    k_cnt = len(picked)
    stakes = np.full(k_cnt, 100, dtype=int)
    rem_units = n_units - k_cnt
    if rem_units > 0:
        weights = np.maximum(picked["ev"].to_numpy(dtype=float) - 0.8, 0.1)
        weights = weights / weights.sum()
        extra = np.floor(weights * rem_units).astype(int)
        stakes += extra * 100
        leftover = n_units - int(stakes.sum() // 100)
        for idx_l in range(leftover):
            stakes[idx_l % k_cnt] += 100

    picked["stake"] = stakes
    picked["expected_return"] = (picked["stake"] * picked["odds"]).round(0).astype(int)
    return picked


# ---------------------------------------------------------
# データ読み込み・状態初期化
# ---------------------------------------------------------
def _load_initial_upcoming() -> dict:
    if os.path.exists(UPCOMING_CSV):
        try:
            df = pd.read_csv(UPCOMING_CSV, dtype={"race_id": str})
            if not df.empty:
                return {
                    "shutuba_df": df,
                    "pair_odds_df": None,
                    "exacta_odds_df": None,
                    "trio_odds_df": None,
                    "has_real_win": True,
                    "has_real_multi": False,
                }
        except Exception:
            pass

    sample_rows = []
    for i in range(1, 13):
        sample_rows.append({
            "race_id": "202605050811",
            "date": datetime.now(JST).strftime("%Y-%m-%d"),
            "venue": "東京",
            "race_title": "サンプルレース（左メニューから開催日取得を実行してください）",
            "start_time": "15:40",
            "surface": "芝",
            "distance": 2000,
            "track_condition": "良",
            "frame_num": min(8, (i - 1) // 2 + 1),
            "horse_num": i,
            "horse_name": f"テストホース{i:02d}",
            "weight_carried": 57.0,
            "horse_weight": 480,
            "weight_diff": 0,
            "jockey_id": "UNKNOWN_J",
            "trainer_id": "UNKNOWN_T",
            "prev1_rank": (i % 7) + 1,
            "prev2_rank": ((i + 2) % 8) + 1,
            "prev3_rank": ((i + 4) % 9) + 1,
            "prev1_time_diff": round(0.1 * i, 2),
            "win_odds": round(2.5 * i, 1),
        })
    return {
        "shutuba_df": pd.DataFrame(sample_rows),
        "pair_odds_df": None,
        "exacta_odds_df": None,
        "trio_odds_df": None,
        "has_real_win": False,
        "has_real_multi": False,
    }


if "race_list" not in st.session_state:
    st.session_state["race_list"] = []
if "current_bundle" not in st.session_state:
    st.session_state["current_bundle"] = _load_initial_upcoming()
if "selected_venue" not in st.session_state:
    init_df = st.session_state["current_bundle"]["shutuba_df"]
    st.session_state["selected_venue"] = str(init_df["venue"].iloc[0]) if "venue" in init_df.columns else "東京"
if "selected_race_num" not in st.session_state:
    init_df = st.session_state["current_bundle"]["shutuba_df"]
    rid_0 = str(init_df["race_id"].iloc[0]) if "race_id" in init_df.columns else "202605050811"
    st.session_state["selected_race_num"] = int(rid_0[10:12]) if len(rid_0) >= 12 and rid_0[10:12].isdigit() else 11


def _select_and_fetch_race(race_id: str, date_str: str):
    if shutuba_scraper is None:
        st.error("06_shutuba_scraper.py が見つかりません。")
        return
    try:
        with st.spinner("出馬表と全券種オッズを取得中..."):
            bundle = shutuba_scraper.fetch_shutuba_and_odds(race_id, date_str)
            st.session_state["current_bundle"] = bundle
            os.makedirs(os.path.dirname(UPCOMING_CSV), exist_ok=True)
            bundle["shutuba_df"].to_csv(UPCOMING_CSV, index=False, encoding="utf-8-sig")
    except Exception as exc:
        st.error(f"レースデータの取得に失敗しました: {exc}")


# ---------------------------------------------------------
# サイドバー（開催日・データ取得・戦略設定）
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 開催日・データ取得")
    default_dt = datetime.now(JST).date()
    target_date = st.date_input("開催日を選択", value=default_dt)
    target_date_str = target_date.strftime("%Y-%m-%d")

    if st.button("開催レース一覧を取得", use_container_width=True, type="primary"):
        if shutuba_scraper is None:
            st.error("06_shutuba_scraper.py が見つかりません。")
        else:
            try:
                with st.spinner(f"{target_date_str} の開催レース一覧を取得中..."):
                    r_list = shutuba_scraper.fetch_race_list_by_date(target_date_str)
                if not r_list:
                    st.warning("指定日のJRA開催レースが見つかりませんでした。土日祝の日付を選択してください。")
                else:
                    st.session_state["race_list"] = r_list
                    venues_avail = list(dict.fromkeys([r["venue"] for r in r_list]))
                    first_v = venues_avail[0]
                    v_races = [r for r in r_list if r["venue"] == first_v]
                    default_r = next((r for r in v_races if r["race_num"] == 11), v_races[-1])
                    st.session_state["selected_venue"] = first_v
                    st.session_state["selected_race_num"] = default_r["race_num"]
                    _select_and_fetch_race(default_r["race_id"], target_date_str)
            except Exception as e:
                st.error(f"一覧取得エラー: {e}")

    race_list = st.session_state["race_list"]
    if race_list:
        cur_v = st.session_state["selected_venue"]
        cur_rn = st.session_state["selected_race_num"]
        cur_item = next(
            (r for r in race_list if r["venue"] == cur_v and r["race_num"] == cur_rn),
            race_list[0],
        )
        labels = [r["label"] for r in race_list]
        cur_idx = labels.index(cur_item["label"])
        chosen_label = st.selectbox("開催レース直接選択", labels, index=cur_idx)
        chosen_obj = next(r for r in race_list if r["label"] == chosen_label)
        if chosen_obj["venue"] != cur_v or chosen_obj["race_num"] != cur_rn:
            st.session_state["selected_venue"] = chosen_obj["venue"]
            st.session_state["selected_race_num"] = chosen_obj["race_num"]
            _select_and_fetch_race(chosen_obj["race_id"], target_date_str)
            st.rerun()

    st.markdown("---")
    st.markdown("### 予想方針・予算設定")
    strategy_choice = st.radio(
        "予想方針",
        ["標準バランス", "穴狙い（1・2番人気軸除外）", "堅実重視"],
        index=0,
    )
    budget_label = st.radio(
        "1レース予算モード",
        ["300円モード", "500円モード", "2000円モード"],
        index=1,
        horizontal=True,
    )
    budget_yen = int(re.sub(r"\D", "", budget_label))

    ai_weight_pct = st.slider("AI独自度 (%)", min_value=20, max_value=100, value=75, step=5)
    min_ev_th = st.slider("推奨最低期待値 (EV)", min_value=0.80, max_value=2.00, value=1.05, step=0.05)


# ---------------------------------------------------------
# 上部ヘッダーバー
# ---------------------------------------------------------
status_cls = "badge-green" if train_status["is_real"] else "badge-warn"
short_strategy = "穴狙い(上位人気除外)" if "穴狙い" in strategy_choice else strategy_choice

st.markdown(
    f"""
<div class="top-navbar">
    <div class="top-navbar-title">JRA KEIBA AI</div>
    <div class="nav-badges">
        <span class="{status_cls}">{train_status["badge"]}</span>
        <span class="badge-slate">方針: {short_strategy}</span>
        <span class="badge-slate">AI独自度: {ai_weight_pct}%</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 写真準拠 2段ナビゲーション（開催馬場ボタン & 1R-12Rボタン）
# ---------------------------------------------------------
race_list = st.session_state["race_list"]
if race_list:
    venues = list(dict.fromkeys([r["venue"] for r in race_list]))
    if st.session_state["selected_venue"] not in venues:
        st.session_state["selected_venue"] = venues[0]
    active_venue = st.session_state["selected_venue"]

    v_cols = st.columns([1] * len(venues) + [2, 1.3])
    for idx_v, v_name in enumerate(venues):
        is_active_v = v_name == active_venue
        if v_cols[idx_v].button(
            v_name,
            key=f"venue_btn_{v_name}",
            type="primary" if is_active_v else "secondary",
            use_container_width=True,
        ):
            st.session_state["selected_venue"] = v_name
            v_races = [r for r in race_list if r["venue"] == v_name]
            target_r = next(
                (r for r in v_races if r["race_num"] == st.session_state["selected_race_num"]),
                v_races[0],
            )
            st.session_state["selected_race_num"] = target_r["race_num"]
            _select_and_fetch_race(target_r["race_id"], target_date_str)
            st.rerun()

    venue_races = [r for r in race_list if r["venue"] == active_venue]
    avail_rnums = {r["race_num"]: r for r in venue_races}
    if st.session_state["selected_race_num"] not in avail_rnums:
        st.session_state["selected_race_num"] = venue_races[0]["race_num"]
    active_rnum = st.session_state["selected_race_num"]

    if v_cols[-1].button("最新更新", key="refresh_odds_btn", use_container_width=True):
        cur_r_obj = avail_rnums[active_rnum]
        _select_and_fetch_race(cur_r_obj["race_id"], target_date_str)
        st.rerun()

    r_cols = st.columns([1.3] + [1] * 12 + [1.3])
    sorted_rnums = sorted(avail_rnums.keys())
    pos_in_list = sorted_rnums.index(active_rnum)

    if r_cols[0].button("← 前R", key="prev_race_btn", disabled=(pos_in_list == 0), use_container_width=True):
        prev_rn = sorted_rnums[pos_in_list - 1]
        st.session_state["selected_race_num"] = prev_rn
        _select_and_fetch_race(avail_rnums[prev_rn]["race_id"], target_date_str)
        st.rerun()

    for r_idx in range(1, 13):
        exists_r = r_idx in avail_rnums
        is_sel_r = r_idx == active_rnum
        if r_cols[r_idx].button(
            f"{r_idx}R",
            key=f"rnum_btn_{r_idx}",
            type="primary" if is_sel_r else "secondary",
            disabled=not exists_r,
            use_container_width=True,
        ):
            st.session_state["selected_race_num"] = r_idx
            _select_and_fetch_race(avail_rnums[r_idx]["race_id"], target_date_str)
            st.rerun()

    if r_cols[13].button(
        "次R →",
        key="next_race_btn",
        disabled=(pos_in_list >= len(sorted_rnums) - 1),
        use_container_width=True,
    ):
        next_rn = sorted_rnums[pos_in_list + 1]
        st.session_state["selected_race_num"] = next_rn
        _select_and_fetch_race(avail_rnums[next_rn]["race_id"], target_date_str)
        st.rerun()


# ---------------------------------------------------------
# 推論・勝率ブレンド・期待値計算
# ---------------------------------------------------------
bundle = st.session_state["current_bundle"]
shutuba_df = bundle["shutuba_df"].copy()

if train_mod is not None and hasattr(train_mod, "create_features"):
    feat_df, _ = train_mod.create_features(shutuba_df, category_maps=category_maps)
else:
    feat_df, _ = _fallback_create_features(shutuba_df, category_maps)

raw_ai_probs = model.predict_proba(feat_df[feature_cols])[:, 1]
raw_ai_probs = np.clip(raw_ai_probs, 1e-5, 1.0)
pure_ai_prob = raw_ai_probs / raw_ai_probs.sum()

win_odds_arr = pd.to_numeric(shutuba_df["win_odds"], errors="coerce").fillna(20.0).to_numpy(dtype=float)
win_odds_arr = np.clip(win_odds_arr, 1.05, 999.9)
mkt_raw = 1.0 / win_odds_arr
mkt_prob = mkt_raw / mkt_raw.sum()

alpha = ai_weight_pct / 100.0
blended_prob = alpha * pure_ai_prob + (1.0 - alpha) * mkt_prob
blended_prob = blended_prob / blended_prob.sum()

quinella_prob, place_prob = compute_harville_probabilities(blended_prob)

shutuba_df["win_odds"] = win_odds_arr
shutuba_df["popularity"] = shutuba_df["win_odds"].rank(method="min").astype(int)
shutuba_df["pure_ai_prob"] = pure_ai_prob
shutuba_df["ai_win_prob"] = blended_prob
shutuba_df["quinella_prob"] = quinella_prob
shutuba_df["place_prob"] = place_prob
shutuba_df["win_ev"] = shutuba_df["ai_win_prob"] * shutuba_df["win_odds"]

shutuba_df["eval_score"] = shutuba_df["ai_win_prob"] * (np.clip(shutuba_df["win_ev"], 0.6, 2.5) ** 0.35)
rank_order = shutuba_df["eval_score"].rank(ascending=False, method="first").astype(int)
marks = []
for r_ord in rank_order:
    if r_ord == 1:
        marks.append("◎")
    elif r_ord == 2:
        marks.append("○")
    elif r_ord == 3:
        marks.append("▲")
    elif r_ord in (4, 5):
        marks.append("△")
    elif r_ord == 6:
        marks.append("☆")
    else:
        marks.append("")
shutuba_df["mark"] = marks

tickets_df = generate_ev_tickets(
    pred_df=shutuba_df,
    pair_odds_df=bundle.get("pair_odds_df"),
    exacta_odds_df=bundle.get("exacta_odds_df"),
    trio_odds_df=bundle.get("trio_odds_df"),
    strategy=strategy_choice,
    budget_yen=budget_yen,
    min_ev=min_ev_th,
)


# ---------------------------------------------------------
# レース情報バナー表示
# ---------------------------------------------------------
first_row = shutuba_df.iloc[0]
rid_str = str(first_row.get("race_id", "202605050811"))
venue_disp = str(first_row.get("venue", "東京"))
rnum_disp = int(rid_str[10:12]) if len(rid_str) >= 12 and rid_str[10:12].isdigit() else 11
start_time_disp = str(first_row.get("start_time", "") or "")
time_badge_html = f'<span class="race-banner-time">{start_time_disp} 発走</span>' if start_time_disp else ""

raw_rtitle = str(first_row.get("race_title", "") or "").strip()
if not raw_rtitle or raw_rtitle == "nan":
    raw_rtitle = f"{venue_disp} {rnum_disp}R"

surface_raw = str(first_row.get("surface", "芝"))
surface_short = "ダ" if "ダ" in surface_raw else "芝"
dist_disp = int(float(first_row.get("distance", 2000)))
cond_disp = str(first_row.get("track_condition", "良"))
n_horses = len(shutuba_df)

if bundle.get("has_real_multi"):
    odds_status_txt = "全券種実オッズ取得済"
elif bundle.get("has_real_win"):
    odds_status_txt = "単勝実オッズ取得済"
else:
    odds_status_txt = "AI期待値算出済"

max_ev_val = float(tickets_df["ev"].max()) if not tickets_df.empty else float(shutuba_df["win_ev"].max())
strong_cnt = int((shutuba_df["win_ev"] >= 1.0).sum())

st.markdown(
    f"""
<div class="race-banner">
    <div>
        <div class="race-banner-sub">{venue_disp} {rnum_disp}R {time_badge_html}</div>
        <div class="race-banner-title">{raw_rtitle}</div>
        <div class="race-banner-meta">
            <span>{surface_short} {dist_disp}m</span>
            <span>{n_horses}頭</span>
            <span class="cond-pill">{surface_short}:{cond_disp}</span>
            <span class="status-pill">{odds_status_txt}</span>
        </div>
    </div>
    <div class="gauge-group">
        <div class="ring-box ring-orange">
            <div class="ring-val-orange">{max_ev_val:.2f}</div>
            <div class="ring-label">最高EV</div>
        </div>
        <div class="ring-box ring-green">
            <div class="ring-val-green">{strong_cnt}</div>
            <div class="ring-label">有力候補</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 推奨買い目（ハービル期待値ベース）
# ---------------------------------------------------------
st.markdown(
    f'<div class="section-heading">AI推奨買い目（{budget_yen:,}円モード / {short_strategy}）</div>',
    unsafe_allow_html=True,
)

bet_filter = st.radio(
    "券種表示フィルタ",
    ["すべて", "単勝", "馬連", "ワイド", "馬単", "三連複"],
    index=0,
    horizontal=True,
)

if tickets_df.empty:
    st.info("条件に合致する推奨買い目が見つかりませんでした。最低期待値（EV）のしきい値を下げて確認してください。")
else:
    view_tickets = tickets_df if bet_filter == "すべて" else tickets_df[tickets_df["bet_type"] == bet_filter]
    if view_tickets.empty:
        st.info(f"選択中の券種（{bet_filter}）では期待値基準を満たす買い目がありません。「すべて」で全券種を確認できます。")
    else:
        disp_t = pd.DataFrame({
            "券種": view_tickets["bet_type"],
            "買い目": view_tickets["combo"],
            "対象馬": view_tickets["horses"],
            "AI的中率": view_tickets["prob"].map(lambda x: f"{x * 100:.1f}%"),
            "オッズ": view_tickets["odds"].map(lambda x: f"{x:.1f}倍"),
            "期待値(EV)": view_tickets["ev"].map(lambda x: f"{x:.2f}"),
            "購入金額": view_tickets["stake"].map(lambda x: f"{int(x):,}円"),
            "想定払戻": view_tickets["expected_return"].map(lambda x: f"{int(x):,}円"),
        })
        st.dataframe(disp_t, use_container_width=True, hide_index=True)

        total_stake = int(view_tickets["stake"].sum())
        max_ret = int(view_tickets["expected_return"].max())
        st.caption(f"表示合計投資額: {total_stake:,}円 / 最大想定払戻: {max_ret:,}円")

    memo_lines = [f"[{venue_disp}{rnum_disp}R {raw_rtitle}] 合計 {int(tickets_df['stake'].sum()):,}円"]
    for _, tr in tickets_df.iterrows():
        memo_lines.append(
            f"{tr['bet_type']} {tr['combo']} ({tr['horses']}) : {int(tr['stake'])}円 [EV {tr['ev']:.2f} / {tr['odds']:.1f}倍]"
        )
    memo_text = "\n".join(memo_lines)

    col_memo, col_save = st.columns([2.2, 1])
    with col_memo:
        with st.expander("スマホ投票用メモ（コピー用テキスト）"):
            st.code(memo_text, language="text")

    with col_save:
        if st.button("この買い目を戦績履歴に保存", use_container_width=True):
            os.makedirs(os.path.dirname(BETS_CSV), exist_ok=True)
            save_rows = tickets_df.copy()
            save_rows["saved_at"] = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
            save_rows["race_id"] = rid_str
            save_rows["venue"] = venue_disp
            save_rows["race_num"] = rnum_disp
            save_rows["race_title"] = raw_rtitle
            save_rows["hit"] = 0
            save_rows["payout"] = 0
            if os.path.exists(BETS_CSV):
                old_b = pd.read_csv(BETS_CSV)
                all_b = pd.concat([old_b, save_rows], ignore_index=True)
            else:
                all_b = save_rows
            all_b.to_csv(BETS_CSV, index=False, encoding="utf-8-sig")
            st.success("買い目を data/bet_history.csv に保存しました。")


# ---------------------------------------------------------
# 出走馬 AI評価・予測勝率一覧
# ---------------------------------------------------------
st.markdown('<div class="section-heading">出走馬 AI予測スコア・単勝期待値一覧</div>', unsafe_allow_html=True)

sort_mode = st.radio(
    "並び順",
    ["馬番順", "AI勝率順", "単勝期待値(EV)順"],
    index=0,
    horizontal=True,
)
if sort_mode == "AI勝率順":
    table_src = shutuba_df.sort_values("ai_win_prob", ascending=False)
elif sort_mode == "単勝期待値(EV)順":
    table_src = shutuba_df.sort_values("win_ev", ascending=False)
else:
    table_src = shutuba_df.sort_values("horse_num", ascending=True)

disp_horses = pd.DataFrame({
    "印": table_src["mark"],
    "枠": table_src["frame_num"].astype(int),
    "馬番": table_src["horse_num"].astype(int),
    "馬名": table_src["horse_name"],
    "騎手": table_src["jockey_id"],
    "斤量": table_src["weight_carried"].map(lambda x: f"{float(x):.1f}"),
    "単勝オッズ": table_src["win_odds"].map(lambda x: f"{float(x):.1f}"),
    "人気": table_src["popularity"].astype(int),
    "AI勝率": table_src["ai_win_prob"].map(lambda x: f"{float(x) * 100:.1f}%"),
    "連対率": table_src["quinella_prob"].map(lambda x: f"{float(x) * 100:.1f}%"),
    "複勝率": table_src["place_prob"].map(lambda x: f"{float(x) * 100:.1f}%"),
    "単勝EV": table_src["win_ev"].map(lambda x: f"{float(x):.2f}"),
})

st.dataframe(disp_horses, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# 戦績・回収率トラッカー
# ---------------------------------------------------------
with st.expander("戦績・回収率トラッカー（保存済み買い目履歴）"):
    if os.path.exists(BETS_CSV):
        try:
            hist_bets = pd.read_csv(BETS_CSV)
            if not hist_bets.empty:
                tot_cost = int(pd.to_numeric(hist_bets["stake"], errors="coerce").fillna(0).sum())
                tot_pay = int(pd.to_numeric(hist_bets["payout"], errors="coerce").fillna(0).sum())
                tot_hits = int((pd.to_numeric(hist_bets["hit"], errors="coerce").fillna(0) > 0).sum())
                roi_pct = (tot_pay / tot_cost * 100.0) if tot_cost > 0 else 0.0
                hit_pct = (tot_hits / len(hist_bets) * 100.0) if len(hist_bets) > 0 else 0.0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("総購入点数", f"{len(hist_bets):,}点")
                m2.metric("的中率", f"{hit_pct:.1f}% ({tot_hits}的中)")
                m3.metric("総投資 / 総払戻", f"{tot_cost:,}円 / {tot_pay:,}円")
                m4.metric("回収率 (ROI)", f"{roi_pct:.1f}%")

                edited_bets = st.data_editor(
                    hist_bets,
                    use_container_width=True,
                    hide_index=True,
                    key="bet_history_editor",
                )
                if st.button("払戻・的中結果を保存", key="save_edited_bets"):
                    edited_bets.to_csv(BETS_CSV, index=False, encoding="utf-8-sig")
                    st.success("戦績データを更新しました。")
            else:
                st.caption("保存された買い目履歴はまだありません。")
        except Exception as exc:
            st.warning(f"履歴ファイルの読み込みエラー: {exc}")
    else:
        st.caption("保存された買い目履歴はまだありません。「この買い目を戦績履歴に保存」ボタンで記録できます。")