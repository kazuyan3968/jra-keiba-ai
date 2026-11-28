from datetime import date, datetime
import importlib
from importlib.util import module_from_spec, spec_from_file_location
import html
import os
from urllib.parse import urlencode
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import harville_calculator
import win5_engine
import horse_detail_view

importlib.reload(harville_calculator)
importlib.reload(win5_engine)
importlib.reload(horse_detail_view)
from harville_calculator import (
    build_budget_mode_tickets,
    build_trio_strategies,
    calculate_exacta_probabilities,
    calculate_horse_place_probs,
    calculate_pair_probabilities,
    calculate_trio_probabilities,
)

# 各種モジュールを読み込み
spec_train = spec_from_file_location("train_model", "02_train_model.py")
train_model = module_from_spec(spec_train)
spec_train.loader.exec_module(train_model)

spec_scraper = spec_from_file_location("shutuba_scraper", "06_shutuba_scraper.py")
shutuba_scraper = module_from_spec(spec_scraper)
spec_scraper.loader.exec_module(shutuba_scraper)

spec_tracker = spec_from_file_location("bet_tracker", "07_bet_tracker.py")
bet_tracker = module_from_spec(spec_tracker)
spec_tracker.loader.exec_module(bet_tracker)

st.set_page_config(page_title="KAZUYAN AI | 期待値＆買い目シミュレーター", layout="wide")

# ==============================================================================
# カスタムCSS（開催馬場ピルボタン・前R/次Rボタン・買い目タイル・出走表・右パネル）
# ==============================================================================
st.markdown(
    """
<style>
.stApp { background-color: #f4f6f9; }
.block-container { padding-top: 3.8rem !important; padding-bottom: 2.5rem; max-width: 1420px; }
div[data-testid="stDeployButton"] { display: none; }

.top-navbar {
    background-color: #16213e; color: #ffffff; padding: 12px 22px; border-radius: 8px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.12); flex-wrap: wrap; gap: 8px;
}
.top-navbar .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: 0.06em; color: #ffffff; }
.top-navbar .nav-badges { display: flex; gap: 8px; align-items: center; font-size: 0.8rem; flex-wrap: wrap; }
.nav-pill { background-color: rgba(255,255,255,0.14); padding: 4px 11px; border-radius: 6px; color: #e2e8f0; font-weight: 600; }
.nav-pill-warn { background-color: #b45309; padding: 4px 11px; border-radius: 6px; color: #ffffff; font-weight: 700; }
.nav-pill-real { background-color: #047857; padding: 4px 11px; border-radius: 6px; color: #ffffff; font-weight: 700; }

.venue-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button,
.race-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    min-height: 40px !important; height: 40px !important; border-radius: 8px !important;
    padding: 4px 10px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    transition: all 0.12s ease-in-out !important;
}
.venue-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="secondary"],
.race-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="secondary"] {
    background-color: #ffffff !important; border: 1.5px solid #cbd5e1 !important; color: #1e293b !important;
}
.venue-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="secondary"]:hover,
.race-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #192442 !important; background-color: #f1f5f9 !important;
}
.venue-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="primary"],
.race-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #192442 !important; border: 1.5px solid #192442 !important; color: #ffffff !important;
}
.venue-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p,
.race-nav-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p {
    font-size: 0.92rem !important; font-weight: 700 !important; white-space: nowrap !important;
}

.race-banner {
    background: linear-gradient(135deg, #4a154b 0%, #611f54 60%, #3b123c 100%);
    color: #ffffff; border-radius: 12px; padding: 18px 24px;
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 6px; margin-bottom: 18px; box-shadow: 0 4px 12px rgba(74, 21, 75, 0.25);
}
.race-title-row { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; flex-wrap: wrap; }
.race-venue-r { font-size: 1.25rem; font-weight: 800; }
.race-time { font-size: 0.88rem; color: #e9d5ff; }
.race-name-main { font-size: 1.45rem; font-weight: 800; margin: 2px 0 8px 0; letter-spacing: 0.02em; }
.race-meta-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 0.85rem; }
.cond-badge { background-color: #fde68a; color: #78350f; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
.status-badge { background-color: #ecfdf5; color: #047857; border: 1px solid #10b981; font-weight: 700; padding: 2px 10px; border-radius: 5px; font-size: 0.78rem; }

.ring-group { display: flex; gap: 14px; }
.ring-circle {
    width: 68px; height: 68px; border-radius: 50%; background: rgba(20, 12, 28, 0.55);
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    box-shadow: inset 0 0 8px rgba(0,0,0,0.4);
}
.ring-orange { border: 3.5px solid #f59e0b; }
.ring-green { border: 3.5px solid #10b981; }
.ring-num-orange { font-size: 1.2rem; font-weight: 800; color: #fbbf24; line-height: 1.1; }
.ring-num-green { font-size: 1.2rem; font-weight: 800; color: #34d399; line-height: 1.1; }
.ring-label { font-size: 0.68rem; color: #f3e8ff; margin-top: 2px; }

.mode-btn-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
    min-height: 84px !important; border-radius: 12px !important; padding: 12px 16px !important;
    transition: all 0.15s ease-in-out !important; box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
}
.mode-btn-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="secondary"] {
    background-color: #ffffff !important; border: 2px solid #cbd5e1 !important; color: #1e293b !important;
}
.mode-btn-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #4a154b !important; background-color: #fdf4ff !important; transform: translateY(-2px);
}
.mode-btn-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #192442 0%, #4a154b 100%) !important;
    border: 2.5px solid #4a154b !important; color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(74, 21, 75, 0.3) !important;
}
.mode-btn-zone + div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button p {
    font-size: 1.45rem !important; font-weight: 800 !important; letter-spacing: 0.04em !important;
}

.shutuba-wrapper {
    background: #ffffff; border-radius: 10px; overflow-x: auto;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; margin-bottom: 20px;
}
.shutuba-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; text-align: center; }
.shutuba-table thead th {
    background-color: #192442; color: #ffffff; font-weight: 600;
    padding: 9px 6px; font-size: 0.78rem; white-space: nowrap; border-bottom: 2px solid #0f172a;
}
.shutuba-table tbody tr { border-bottom: 1px solid #e2e8f0; transition: background-color 0.15s; }
.shutuba-table tbody tr:hover { filter: brightness(0.97); }
.row-rank-1 { background-color: #fdf2f2; }
.row-rank-2 { background-color: #fff8eb; }
.row-rank-3 { background-color: #f8fafc; }
.row-selected { background-color: #eff6ff !important; box-shadow: inset 4px 0 0 #2563eb; }
.shutuba-table td { padding: 8px 6px; vertical-align: middle; }

.waku-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%; font-weight: 800;
    font-size: 0.82rem; box-shadow: 0 1px 2px rgba(0,0,0,0.18);
}
.waku-1 { background: #ffffff; color: #111827; border: 1.5px solid #9ca3af; }
.waku-2 { background: #1f2937; color: #ffffff; }
.waku-3 { background: #dc2626; color: #ffffff; }
.waku-4 { background: #2563eb; color: #ffffff; }
.waku-5 { background: #eab308; color: #111827; }
.waku-6 { background: #16a34a; color: #ffffff; }
.waku-7 { background: #f97316; color: #ffffff; }
.waku-8 { background: #ec4899; color: #ffffff; }

.umaban-txt { font-weight: 800; font-size: 1.0rem; }
.umaban-top { color: #dc2626; }
.mark-badge { font-weight: 900; font-size: 1.02rem; color: #dc2626; }
.pop-tag { font-size: 0.70rem; color: #2563eb; font-weight: 600; line-height: 1.1; }
.odds-val { font-size: 0.98rem; font-weight: 800; color: #111827; line-height: 1.2; }

.horse-cell { text-align: left !important; padding-left: 10px !important; white-space: nowrap; }
.horse-title { font-weight: 800; font-size: 0.93rem; color: #1d4ed8; line-height: 1.25; }
.horse-link { color: #1d4ed8 !important; text-decoration: none !important; cursor: pointer; }
.horse-link:hover { text-decoration: underline !important; color: #1e40af !important; }
.jockey-sub { font-size: 0.74rem; color: #64748b; margin-top: 2px; }

.sig-axis { background-color: #dc2626; color: #ffffff; font-weight: 800; font-size: 0.74rem; padding: 3px 10px; border-radius: 5px; display: inline-block; white-space: nowrap; }
.sig-value { background-color: #fff7ed; color: #c2410c; border: 1px solid #f97316; font-weight: 800; font-size: 0.74rem; padding: 2px 8px; border-radius: 5px; display: inline-block; white-space: nowrap; }
.sig-himo { background-color: #ffffff; color: #2563eb; border: 1px solid #93c5fd; font-weight: 700; font-size: 0.74rem; padding: 2px 8px; border-radius: 5px; display: inline-block; white-space: nowrap; }
.sig-wait { background-color: #f8fafc; color: #64748b; border: 1px solid #cbd5e1; font-weight: 600; font-size: 0.74rem; padding: 2px 8px; border-radius: 5px; display: inline-block; white-space: nowrap; }

.conf-s { color: #dc2626; font-weight: 900; font-size: 1.02rem; }
.conf-a { color: #dc2626; font-weight: 800; font-size: 0.98rem; }
.conf-b { color: #d97706; font-weight: 800; font-size: 0.92rem; }
.conf-none { color: #94a3b8; font-weight: 600; }

.ability-wrap { display: inline-flex; align-items: center; gap: 6px; justify-content: center; }
.ability-num { font-weight: 800; font-size: 0.88rem; min-width: 34px; text-align: right; }
.bar-bg { width: 48px; height: 7px; background-color: #e2e8f0; border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }

.prob-hot { color: #dc2626; font-weight: 800; }
.prob-warm { color: #ea580c; font-weight: 700; }
.prob-norm { color: #1e293b; font-weight: 600; }

.bet-card {
    background: #ffffff; border-left: 5px solid #4a154b; border-radius: 8px;
    padding: 14px 18px; margin-top: 10px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.bet-card-title { font-size: 1.04rem; font-weight: 800; color: #192442; margin-bottom: 8px; }
.horse-chip {
    display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px;
    padding: 3px 10px; margin: 2px 5px 2px 0; font-weight: 700; font-size: 0.86rem; color: #0f172a;
}
.horse-chip-axis { background: #fef2f2; border-color: #fca5a5; color: #b91c1c; }

.help-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.help-card h4 { color: #192442; margin-top: 0; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 2px solid #4a154b; font-weight: 800; }
.help-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 8px; }
.help-table th { background-color: #f1f5f9; color: #192442; text-align: left; padding: 8px 12px; border: 1px solid #cbd5e1; width: 24%; }
.help-table td { padding: 8px 12px; border: 1px solid #e2e8f0; color: #334155; line-height: 1.5; }
.formula-box {
    background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #192442;
    padding: 10px 14px; border-radius: 6px; font-family: Consolas, monospace; font-size: 0.88rem; color: #0f172a; margin: 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact(model_mtime: float):
    return joblib.load("models/keiba_lgbm.pkl")


@st.cache_data(ttl=180, show_spinner=False)
def _cached_fetch_race_list(date_str: str) -> list[dict]:
    return shutuba_scraper.fetch_race_list_by_date(date_str)


@st.cache_data(ttl=180, show_spinner=False)
def _cached_fetch_shutuba_and_odds(race_id: str, date_str: str) -> dict:
    return shutuba_scraper.fetch_shutuba_and_odds(race_id=race_id, date_str=date_str)


@st.cache_data(ttl=30)
def get_training_data_status(_artifact: dict | None = None, model_mtime: float = 0.0) -> dict:
    hist_path = "data/historical_races.csv"
    if os.path.exists(hist_path):
        try:
            df_h = pd.read_csv(hist_path, usecols=["race_id", "date", "horse_name"])
            sample_ratio = df_h["horse_name"].astype(str).str.startswith("サンプルホース").mean()
            n_races = int(df_h["race_id"].nunique())
            min_d = str(df_h["date"].min())
            max_d = str(df_h["date"].max())
            if sample_ratio > 0.5:
                return {
                    "is_real": False,
                    "badge": "学習元: テスト用サンプルデータ",
                    "detail": f"現在は 01_data_loader.py で生成した仮想データ（{n_races:,}レース / {min_d} 〜 {max_d}）で学習しています。",
                }
            return {
                "is_real": True,
                "badge": f"学習元: JRA実データ ({n_races:,}R)",
                "detail": f"実際のJRA過去レース結果（{n_races:,}レース / 期間: {min_d} 〜 {max_d}）を使用して学習しています。",
            }
        except Exception:
            pass

    if _artifact and isinstance(_artifact, dict):
        jockeys = [str(j) for j in _artifact.get("category_maps", {}).get("jockey_id", [])]
        real_jockeys = [
            j for j in jockeys
            if not j.startswith("JOCKEY_") and j not in ("UNKNOWN_J", "UNKNOWN", "nan", "")
        ]
        n_r = _artifact.get("n_races")
        if len(real_jockeys) >= 20 or (n_r is not None and int(n_r) > 0):
            badge_txt = f"学習元: JRA実データ ({int(n_r):,}R)" if n_r else "学習元: JRA実データモデル適用中"
            detail_txt = (
                f"実際のJRA過去レース結果（{int(n_r):,}レース）で学習済みのモデル（models/keiba_lgbm.pkl）を使用して予測しています。"
                if n_r
                else "実際のJRA過去レース結果で学習済みのモデル（models/keiba_lgbm.pkl）を使用して予測しています。"
            )
            return {
                "is_real": True,
                "badge": badge_txt,
                "detail": detail_txt,
            }

    return {
        "is_real": False,
        "badge": "学習データ未確認",
        "detail": "models/keiba_lgbm.pkl または data/historical_races.csv を確認してください。",
    }


def load_race_into_session(
    race_item: dict,
    date_str: str,
    force_refresh: bool = False,
    keep_detail_horse: bool = False,
):
    """指定されたレースの出馬表と全オッズを取得してセッションに反映"""
    if force_refresh:
        _cached_fetch_shutuba_and_odds.clear()
        bundle = shutuba_scraper.fetch_shutuba_and_odds(
            race_id=race_item["race_id"],
            date_str=date_str,
        )
    else:
        bundle = _cached_fetch_shutuba_and_odds(
            race_id=race_item["race_id"],
            date_str=date_str,
        )
    scraped_df = bundle["shutuba_df"].copy()
    os.makedirs("data", exist_ok=True)
    scraped_df.to_csv("data/upcoming_race.csv", index=False)

    st.session_state["active_race_df"] = scraped_df
    st.session_state["auto_pair_odds_df"] = bundle["pair_odds_df"]
    st.session_state["auto_exacta_odds_df"] = bundle["exacta_odds_df"]
    st.session_state["auto_trio_odds_df"] = bundle["trio_odds_df"]
    st.session_state["selected_venue"] = race_item["venue"]
    st.session_state["selected_race_id"] = race_item["race_id"]
    st.session_state["odds_status_label"] = (
        "全券種実オッズ取得済"
        if bundle["has_real_multi"]
        else ("単勝実オッズ取得済" if bundle["has_real_win"] else "発売前仮オッズ")
    )
    if not keep_detail_horse:
        st.session_state["selected_detail_horse"] = None


def _get_query_param(key: str, default: str = "") -> str:
    try:
        val = st.query_params.get(key, default)
        if isinstance(val, list):
            return str(val[0]) if val else default
        return str(val) if val is not None else default
    except Exception:
        return default


# ==============================================================================
# 馬名クリック（同一画面インライン表示）のクエリパラメータ復元処理
# ==============================================================================
q_horse = _get_query_param("horse", "")
if q_horse.isdigit():
    st.session_state["selected_detail_horse"] = int(q_horse)

    q_rid = _get_query_param("race_id", "")
    q_date = _get_query_param("date", "")
    q_mode = _get_query_param("mode", "")
    q_style = _get_query_param("style", "")
    q_aiw = _get_query_param("ai_w", "")
    q_ev = _get_query_param("ev_th", "")
    q_ex1 = _get_query_param("ex1", "")

    if q_mode in ("200", "300", "500", "2000"):
        st.session_state["selected_budget_mode"] = q_mode
    if q_style in ("balanced", "solid", "upset"):
        st.session_state["_init_style_mode"] = q_style
    if q_aiw:
        try:
            st.session_state["_init_ai_weight"] = float(q_aiw)
        except ValueError:
            pass
    if q_ev:
        try:
            st.session_state["_init_ev_threshold"] = float(q_ev)
        except ValueError:
            pass
    if q_ex1 in ("0", "1"):
        st.session_state["_init_exclude_fav1"] = (q_ex1 == "1")

    if q_date:
        st.session_state["fetched_race_date"] = q_date
        if not st.session_state.get("fetched_race_list"):
            try:
                r_list_cached = _cached_fetch_race_list(q_date)
                if r_list_cached:
                    st.session_state["fetched_race_list"] = r_list_cached
            except Exception:
                pass

    if q_rid and st.session_state.get("selected_race_id") != q_rid:
        r_list_now = st.session_state.get("fetched_race_list", [])
        hit_item = next((r for r in r_list_now if r["race_id"] == q_rid), None)
        if hit_item is None:
            v_map = {
                "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
                "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
            }
            v_nm = v_map.get(q_rid[4:6], "東京") if len(q_rid) >= 6 else "東京"
            r_nm = int(q_rid[-2:]) if len(q_rid) >= 2 and q_rid[-2:].isdigit() else 11
            hit_item = {"race_id": q_rid, "venue": v_nm, "race_num": r_nm}
        try:
            load_race_into_session(
                hit_item,
                q_date or date.today().strftime("%Y-%m-%d"),
                force_refresh=False,
                keep_detail_horse=True,
            )
        except Exception:
            pass

    try:
        st.query_params.clear()
    except Exception:
        pass


model_path = "models/keiba_lgbm.pkl"
model_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
artifact = load_artifact(model_mtime)
model = artifact["model"]
feature_cols = artifact["feature_cols"]
category_maps = artifact["category_maps"]
train_status = get_training_data_status(artifact, model_mtime)

# ==============================================================================
# サイドバー：開催日＆レース取得・設定
# ==============================================================================
st.sidebar.header("1. 開催日・データ取得")
data_source = st.sidebar.radio(
    "データ取得方法",
    ["WebからJRA出馬表・オッズを自動取得", "ローカル / CSVアップロード"],
)

default_date_val = date.today()
if st.session_state.get("fetched_race_date"):
    try:
        default_date_val = datetime.strptime(st.session_state["fetched_race_date"], "%Y-%m-%d").date()
    except Exception:
        default_date_val = date.today()

uploaded_file = None
if data_source == "WebからJRA出馬表・オッズを自動取得":
    target_date = st.sidebar.date_input("開催日を選択", value=default_date_val)
    target_date_str = target_date.strftime("%Y-%m-%d")

    if st.sidebar.button("開催レース一覧を取得", width="stretch"):
        with st.spinner("開催レース一覧を取得中..."):
            try:
                _cached_fetch_race_list.clear()
                races_found = _cached_fetch_race_list(target_date_str)
                st.session_state["fetched_race_list"] = races_found
                st.session_state["fetched_race_date"] = target_date_str
                if not races_found:
                    st.sidebar.warning("指定日のJRA開催レースが見つかりませんでした。土日祝の日付を指定してください。")
                else:
                    venues_avail = list(dict.fromkeys(r["venue"] for r in races_found))
                    first_v = venues_avail[0]
                    v_races = [r for r in races_found if r["venue"] == first_v]
                    default_r = next((r for r in v_races if r["race_num"] == 11), v_races[0])
                    load_race_into_session(default_r, target_date_str)
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"一覧取得エラー: {e}")

    race_list = st.session_state.get("fetched_race_list", [])
    if race_list:
        label_to_item = {item["label"]: item for item in race_list}
        current_rid = st.session_state.get("selected_race_id")
        labels_list = list(label_to_item.keys())
        default_idx = 0
        for idx_l, item in enumerate(race_list):
            if item["race_id"] == current_rid:
                default_idx = idx_l
                break

        selected_label = st.sidebar.selectbox("対象レース（または上のボタンで選択）", labels_list, index=default_idx)
        if st.sidebar.button("選択レースの出馬表とオッズを取得", type="primary", width="stretch"):
            sel_item = label_to_item[selected_label]
            with st.spinner(f"{sel_item['venue']} {sel_item['race_num']}R を取得中..."):
                try:
                    load_race_into_session(
                        sel_item,
                        st.session_state.get("fetched_race_date", target_date_str),
                    )
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"取得エラー: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("出馬表CSV（任意）", type=["csv"])

st.sidebar.header("2. 狙い方・バランス調整")
style_options = [
    "標準バランス（実力軸＋妙味中穴）",
    "堅実重視（的中率優先）",
    "穴狙い特化（高配当EV優先）",
]
style_code_to_idx = {"balanced": 0, "solid": 1, "upset": 2}
init_style_idx = style_code_to_idx.get(st.session_state.pop("_init_style_mode", "balanced"), 0)

style_label = st.sidebar.radio(
    "狙い方スタイル",
    style_options,
    index=init_style_idx,
)
style_map = {
    "標準バランス（実力軸＋妙味中穴）": "balanced",
    "堅実重視（的中率優先）": "solid",
    "穴狙い特化（高配当EV優先）": "upset",
}
selected_style_mode = style_map[style_label]

init_ai_w = float(st.session_state.pop("_init_ai_weight", 0.75))
ai_weight = st.sidebar.slider(
    "AI独自度（低いほど実オッズの実力評価を加味）",
    min_value=0.40,
    max_value=1.00,
    value=float(np.clip(init_ai_w, 0.40, 1.00)),
    step=0.05,
)
init_ex1 = bool(st.session_state.pop("_init_exclude_fav1", False))
exclude_fav1 = st.sidebar.checkbox("単勝1番人気を軸・BOXから除外", value=init_ex1)

init_ev_th = float(st.session_state.pop("_init_ev_threshold", 1.15))
ev_threshold = st.sidebar.slider(
    "期待値（EV）下限",
    min_value=0.80,
    max_value=2.50,
    value=float(np.clip(init_ev_th, 0.80, 2.50)),
    step=0.05,
)

with st.sidebar.expander("詳細フィルタ・手動オッズCSV"):
    pair_odds_file = st.file_uploader("馬連・ワイド実オッズCSV", type=["csv"])
    trio_odds_file = st.file_uploader("三連複実オッズCSV", type=["csv"])
    min_win_prob = st.slider("単勝 最低予測勝率（%）", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    min_pair_prob = st.slider("馬連・ワイド 最低的中率（%）", min_value=0.0, max_value=15.0, value=1.5, step=0.5)
    min_trio_prob = st.slider("三連複 最低的中率（%）", min_value=0.0, max_value=5.0, value=0.4, step=0.1)
    max_tickets = st.slider("詳細タブ 最大買い目点数", min_value=3, max_value=25, value=10, step=1)
    budget = st.number_input("詳細タブ 自由予算（円）", min_value=1000, max_value=100000, value=10000, step=1000)

# 対象出馬表の決定
if uploaded_file is not None:
    race_df = pd.read_csv(uploaded_file)
elif "active_race_df" in st.session_state:
    race_df = st.session_state["active_race_df"].copy()
else:
    race_df = pd.read_csv("data/upcoming_race.csv")

pair_odds_df = pd.read_csv(pair_odds_file) if pair_odds_file is not None else st.session_state.get("auto_pair_odds_df")
exacta_odds_df = st.session_state.get("auto_exacta_odds_df")
trio_odds_df = pd.read_csv(trio_odds_file) if trio_odds_file is not None else st.session_state.get("auto_trio_odds_df")

# 1. 単勝予測確率の算出
feat_df, _ = train_model.create_features(race_df, category_maps=category_maps)
raw_probs = model.predict_proba(feat_df[feature_cols])[:, 1]
ai_pure_prob = raw_probs / raw_probs.sum()

mkt_raw = 0.80 / np.clip(race_df["win_odds"].to_numpy(dtype=float), 1.1, 500.0)
mkt_prob = mkt_raw / mkt_raw.sum()

blended_prob = (ai_weight * ai_pure_prob) + ((1.0 - ai_weight) * mkt_prob)
race_df["win_prob"] = blended_prob / blended_prob.sum()

# 2. ハービル公式による全券種確率＆期待値の算出
race_df = calculate_horse_place_probs(race_df, prob_col="win_prob")
pairs_df = calculate_pair_probabilities(race_df, prob_col="win_prob", pair_odds_df=pair_odds_df)
exacta_df = calculate_exacta_probabilities(race_df, prob_col="win_prob", exacta_odds_df=exacta_odds_df)
trio_df = calculate_trio_probabilities(race_df, prob_col="win_prob", trio_odds_df=trio_odds_df)

budget_modes = build_budget_mode_tickets(
    horses_df=race_df,
    pairs_df=pairs_df,
    exacta_df=exacta_df,
    trio_df=trio_df,
    exclude_fav1=exclude_fav1,
    style_mode=selected_style_mode,
)

# 200円モード（期待値×的中率バランス最上位の厳選2点）を生成
df_300_src = budget_modes["mode_300"]["tickets_df"]
df_200_tickets = df_300_src.head(2).copy().reset_index(drop=True)
if not df_200_tickets.empty and "購入額(円)" in df_200_tickets.columns:
    df_200_tickets["購入額(円)"] = 100
    cost_200 = int(df_200_tickets["購入額(円)"].sum())
else:
    cost_200 = len(df_200_tickets) * 100
budget_modes["mode_200"] = {
    "tickets_df": df_200_tickets,
    "total_cost": cost_200,
}

trio_strategies = build_trio_strategies(
    trio_df=trio_df,
    horses_df=race_df,
    ev_threshold=ev_threshold,
    min_prob_pct=min_trio_prob,
    max_single_tickets=max_tickets,
    max_followers=min(max_tickets, 8),
    exclude_fav1=exclude_fav1,
)

# 人気順・期待値・偏差値風能力スコアの付与
race_df["popularity"] = race_df["win_odds"].rank(method="min", ascending=True).astype(int)
race_df["単勝期待値(EV)"] = (race_df["pred_win_pct"] / 100.0) * race_df["win_odds"]

p_mean = race_df["pred_win_pct"].mean()
p_std = max(race_df["pred_win_pct"].std(), 1e-6)
race_df["ability_score"] = np.clip(50.0 + ((race_df["pred_win_pct"] - p_mean) / p_std) * 8.5, 35.0, 78.0).round(1)

race_df = race_df.sort_values("pred_win_pct", ascending=False).reset_index(drop=True)
race_df["ai_rank"] = np.arange(1, len(race_df) + 1)

marks = ["◎", "○", "▲", "△", "☆"]
race_df["印"] = [marks[i] if i < len(marks) else "--" for i in range(len(race_df))]


def get_signal_and_conf(row) -> tuple[str, str]:
    rank = int(row["ai_rank"])
    ev = float(row["単勝期待値(EV)"])
    win_p = float(row["pred_win_pct"])

    if rank == 1 and not (exclude_fav1 and int(row["popularity"]) == 1):
        sig = '<span class="sig-axis">軸</span>'
    elif ev >= ev_threshold and win_p >= 4.0:
        sig = '<span class="sig-value">穴注目</span>'
    elif rank <= 6 or float(row["pred_place_pct"]) >= 25.0:
        sig = '<span class="sig-himo">紐候補</span>'
    else:
        sig = '<span class="sig-wait">様子見</span>'

    if rank == 1 and win_p >= 22.0:
        conf = '<span class="conf-s">S</span>'
    elif rank == 1 or (rank == 2 and win_p >= 14.0):
        conf = '<span class="conf-a">A</span>'
    elif rank <= 3 and win_p >= 9.0:
        conf = '<span class="conf-b">B</span>'
    else:
        conf = '<span class="conf-none">-</span>'
    return sig, conf


sigs, confs = [], []
for _, r in race_df.iterrows():
    s_html, c_html = get_signal_and_conf(r)
    sigs.append(s_html)
    confs.append(c_html)
race_df["signal_html"] = sigs
race_df["conf_html"] = confs

# ==============================================================================
# トップナビゲーションバー（ブランド名: KAZUYAN AI）
# ==============================================================================
data_pill_cls = "nav-pill-real" if train_status["is_real"] else "nav-pill-warn"
style_short_badge = style_label.split("（")[0]
badge_esc = html.escape(train_status["badge"])
style_esc = html.escape(style_short_badge)
navbar_html = (
    '<div class="top-navbar">'
    '<div class="brand">KAZUYAN AI</div>'
    '<div class="nav-badges">'
    f'<span class="{data_pill_cls}">{badge_esc}</span>'
    f'<span class="nav-pill">方針: {style_esc}</span>'
    f'<span class="nav-pill">AI独自度: {int(ai_weight * 100)}%</span>'
    '</div>'
    '</div>'
)
st.markdown(navbar_html, unsafe_allow_html=True)

tab_main, tab_win5, tab_stats, tab_help = st.tabs([
    "レース予想・買い目",
    "WIN5 期待値予想",
    "戦績・回収率",
    "ヘルプ・算出ロジック解説",
])

# ==============================================================================
# タブ1：メインの予想・買い目画面
# ==============================================================================
with tab_main:
    r_info = race_df.iloc[0]
    race_id_str = str(r_info.get("race_id", "202605050811"))
    race_num_int = int(race_id_str[-2:]) if len(race_id_str) >= 2 and race_id_str[-2:].isdigit() else 11
    race_num_label = f"{race_num_int}R"
    current_venue = str(r_info["venue"])
    current_date_str = st.session_state.get("fetched_race_date", str(r_info["date"]))

    # ==========================================================================
    # 開催馬場選択ボタンバー ＋ レース移動ボタンバー
    # ==========================================================================
    fetched_list = st.session_state.get("fetched_race_list", [])
    if fetched_list:
        venues_in_day = list(dict.fromkeys(item["venue"] for item in fetched_list))
        active_venue = st.session_state.get("selected_venue", current_venue)
        if active_venue not in venues_in_day:
            active_venue = venues_in_day[0]

        st.markdown('<div class="venue-nav-zone"></div>', unsafe_allow_html=True)
        v_cols = st.columns(len(venues_in_day) + 1)
        for idx_v, v_name in enumerate(venues_in_day):
            with v_cols[idx_v]:
                is_active_v = (v_name == active_venue)
                if st.button(
                    v_name,
                    key=f"btn_venue_{v_name}",
                    type="primary" if is_active_v else "secondary",
                    width="stretch",
                ):
                    st.session_state["selected_venue"] = v_name
                    v_races = [r for r in fetched_list if r["venue"] == v_name]
                    target_r = next(
                        (r for r in v_races if r["race_num"] == race_num_int),
                        next((r for r in v_races if r["race_num"] == 11), v_races[0]),
                    )
                    with st.spinner(f"{v_name} {target_r['race_num']}R を読み込み中..."):
                        try:
                            load_race_into_session(target_r, current_date_str)
                            st.rerun()
                        except Exception as e:
                            st.error(f"取得エラー: {e}")

        with v_cols[-1]:
            if st.button("最新更新", key="btn_refresh_odds", type="secondary", width="stretch"):
                cur_item = next((r for r in fetched_list if r["race_id"] == race_id_str), None)
                if cur_item:
                    with st.spinner("最新オッズを再取得中..."):
                        try:
                            load_race_into_session(cur_item, current_date_str, force_refresh=True, keep_detail_horse=True)
                            st.rerun()
                        except Exception as e:
                            st.error(f"更新エラー: {e}")

        venue_races = sorted(
            [r for r in fetched_list if r["venue"] == active_venue],
            key=lambda x: x["race_num"],
        )
        if venue_races:
            cur_idx_in_v = next(
                (i for i, r in enumerate(venue_races) if r["race_id"] == race_id_str),
                0,
            )
            st.markdown('<div class="race-nav-zone"></div>', unsafe_allow_html=True)
            nav_cols = st.columns([1.4] + [1.0] * len(venue_races) + [1.4])

            with nav_cols[0]:
                prev_disabled = (cur_idx_in_v <= 0)
                if st.button("← 前R", key="btn_prev_race", disabled=prev_disabled, width="stretch"):
                    target_r = venue_races[cur_idx_in_v - 1]
                    with st.spinner(f"{active_venue} {target_r['race_num']}R を取得中..."):
                        try:
                            load_race_into_session(target_r, current_date_str)
                            st.rerun()
                        except Exception as e:
                            st.error(f"取得エラー: {e}")

            for idx_r, r_item in enumerate(venue_races):
                with nav_cols[idx_r + 1]:
                    is_cur_r = (r_item["race_id"] == race_id_str)
                    if st.button(
                        f"{r_item['race_num']}R",
                        key=f"btn_rnum_{r_item['race_id']}",
                        type="primary" if is_cur_r else "secondary",
                        width="stretch",
                    ):
                        if not is_cur_r:
                            with st.spinner(f"{active_venue} {r_item['race_num']}R を取得中..."):
                                try:
                                    load_race_into_session(r_item, current_date_str)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"取得エラー: {e}")

            with nav_cols[-1]:
                next_disabled = (cur_idx_in_v >= len(venue_races) - 1)
                if st.button("次R →", key="btn_next_race", disabled=next_disabled, width="stretch"):
                    target_r = venue_races[cur_idx_in_v + 1]
                    with st.spinner(f"{active_venue} {target_r['race_num']}R を取得中..."):
                        try:
                            load_race_into_session(target_r, current_date_str)
                            st.rerun()
                        except Exception as e:
                            st.error(f"取得エラー: {e}")
    else:
        st.markdown('<div class="venue-nav-zone"></div>', unsafe_allow_html=True)
        q_c1, q_c2 = st.columns([2, 3])
        with q_c1:
            if st.button("本日の開催馬場一覧を読み込む", type="primary", width="stretch"):
                today_s = date.today().strftime("%Y-%m-%d")
                with st.spinner("開催馬場一覧を取得中..."):
                    try:
                        races_found = _cached_fetch_race_list(today_s)
                        st.session_state["fetched_race_list"] = races_found
                        st.session_state["fetched_race_date"] = today_s
                        if races_found:
                            venues_avail = list(dict.fromkeys(r["venue"] for r in races_found))
                            v_races = [r for r in races_found if r["venue"] == venues_avail[0]]
                            default_r = next((r for r in v_races if r["race_num"] == 11), v_races[0])
                            load_race_into_session(default_r, today_s)
                            st.rerun()
                        else:
                            st.info("本日はJRA開催がありません。左サイドバーから土日祝の日付を選択してください。")
                    except Exception as e:
                        st.error(f"取得エラー: {e}")
        with q_c2:
            st.caption("※ 過去日や週末のレースを選ぶ場合は、左上の「>」からサイドバーを開いて開催日を選択してください。")

    # ==========================================================================
    # レース情報バナー
    # ==========================================================================
    surface_short = "ダ" if "ダ" in str(r_info["surface"]) else "芝"
    race_title_display = (
        str(r_info["race_title"])
        if "race_title" in race_df.columns and pd.notna(r_info["race_title"]) and str(r_info["race_title"]).strip() != ""
        else f"{r_info['venue']} {surface_short}{int(r_info['distance'])}m"
    )
    start_time_display = (
        f"{r_info['start_time']} 発走"
        if "start_time" in race_df.columns and pd.notna(r_info["start_time"]) and str(r_info["start_time"]).strip() != ""
        else f"{r_info['date']} 開催"
    )
    odds_badge_text = st.session_state.get("odds_status_label", "AI期待値算出済")

    race_meta = {
        "race_id": race_id_str,
        "date": str(r_info["date"]),
        "venue": str(r_info["venue"]),
        "race_num": race_num_int,
        "race_title": race_title_display,
    }

    max_ev_val = float(race_df["単勝期待値(EV)"].max())
    active_candidates = int((race_df["ai_rank"] <= 6).sum())

    venue_esc = html.escape(str(r_info["venue"]))
    time_esc = html.escape(start_time_display)
    rtitle_esc = html.escape(race_title_display)
    cond_esc = html.escape(str(r_info["track_condition"]))
    odds_b_esc = html.escape(odds_badge_text)
    dist_i = int(r_info["distance"])

    banner_html = (
        '<div class="race-banner">'
        '<div>'
        '<div class="race-title-row">'
        f'<span class="race-venue-r">{venue_esc} {race_num_label}</span>'
        f'<span class="race-time">{time_esc}</span>'
        '</div>'
        f'<div class="race-name-main">{rtitle_esc}</div>'
        '<div class="race-meta-row">'
        f'<span>{surface_short} {dist_i}m</span>'
        f'<span>{len(race_df)}頭</span>'
        f'<span class="cond-badge">{surface_short}:{cond_esc}</span>'
        f'<span class="status-badge">{odds_b_esc}</span>'
        '</div>'
        '</div>'
        '<div class="ring-group">'
        '<div class="ring-circle ring-orange">'
        f'<span class="ring-num-orange">{max_ev_val:.2f}</span>'
        '<span class="ring-label">最高EV</span>'
        '</div>'
        '<div class="ring-circle ring-green">'
        f'<span class="ring-num-green">{active_candidates}</span>'
        '<span class="ring-label">有力候補</span>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    def render_horse_chips(nums: list[int], df_horses: pd.DataFrame, is_axis: bool = False) -> str:
        num_to_name = dict(zip(df_horses["horse_num"].astype(int), df_horses["horse_name"]))
        cls = "horse-chip horse-chip-axis" if is_axis else "horse-chip"
        chips = []
        for n in nums:
            nm_esc = html.escape(str(num_to_name.get(n, "")))
            chips.append(f'<span class="{cls}">{n:02d} {nm_esc}</span>')
        return "".join(chips)

    # ==========================================================================
    # 買い目モード選択パネル（200円 / 300円 / 500円 / 2000円）
    # ==========================================================================
    st.markdown("### 買い目モード選択")

    if "selected_budget_mode" not in st.session_state:
        st.session_state["selected_budget_mode"] = "500"

    current_mode = st.session_state["selected_budget_mode"]

    st.markdown('<div class="mode-btn-zone"></div>', unsafe_allow_html=True)
    btn_col0, btn_col1, btn_col2, btn_col3 = st.columns(4)
    with btn_col0:
        if st.button(
            "200円",
            key="btn_mode_200",
            type="primary" if current_mode == "200" else "secondary",
            width="stretch",
        ):
            st.session_state["selected_budget_mode"] = "200"
            st.rerun()

    with btn_col1:
        if st.button(
            "300円",
            key="btn_mode_300",
            type="primary" if current_mode == "300" else "secondary",
            width="stretch",
        ):
            st.session_state["selected_budget_mode"] = "300"
            st.rerun()

    with btn_col2:
        if st.button(
            "500円",
            key="btn_mode_500",
            type="primary" if current_mode == "500" else "secondary",
            width="stretch",
        ):
            st.session_state["selected_budget_mode"] = "500"
            st.rerun()

    with btn_col3:
        if st.button(
            "2000円",
            key="btn_mode_2000",
            type="primary" if current_mode == "2000" else "secondary",
            width="stretch",
        ):
            st.session_state["selected_budget_mode"] = "2000"
            st.rerun()

    current_mode = st.session_state["selected_budget_mode"]

    if current_mode == "200":
        m200 = budget_modes["mode_200"]
        df_200 = m200["tickets_df"]
        card_200 = (
            '<div class="bet-card">'
            f'<div class="bet-card-title">200円モード：期待値(EV)と的中率バランス超厳選2点（計 {len(df_200)}点 × 100円 ＝ {m200["total_cost"]}円）</div>'
            f'<div style="font-size:0.86rem; color:#475569;">現在の狙い方スタイル（{style_esc}）に基づき、全券種から最優先の2点だけを厳選抽出しています。</div>'
            '</div>'
        )
        st.markdown(card_200, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("合計購入金額", f"{m200['total_cost']} 円（{len(df_200)}点）")
        c2.metric("平均期待値(EV)", f"{df_200['期待値(EV)'].mean():.2f}" if not df_200.empty else "-")
        c3.metric("最大想定払戻", f"{df_200['想定払戻(円)'].max():,} 円" if not df_200.empty else "-")
        st.dataframe(
            df_200[["券種", "組み合わせ", "対象馬名", "オッズ", "的中率(%)", "期待値(EV)", "購入額(円)", "想定払戻(円)"]]
            .style.format({"オッズ": "{:.1f}", "的中率(%)": "{:.2f}%", "期待値(EV)": "{:.2f}"}),
            width="stretch",
            hide_index=True,
        )

    elif current_mode == "500":
        m500 = budget_modes["mode_500"]
        df_500 = m500["tickets_df"]
        axis_html = render_horse_chips(m500["axis"], race_df, is_axis=True)
        foll_html = render_horse_chips(m500["followers"], race_df, is_axis=False)
        card_500 = (
            '<div class="bet-card">'
            f'<div class="bet-card-title">500円モード：3連複 2頭軸5頭流し（計 {len(df_500)}点 × 100円 ＝ {m500["total_cost"]}円）</div>'
            f'<div style="margin-bottom:6px;"><strong>軸2頭：</strong> {axis_html}</div>'
            f'<div><strong>相手5頭：</strong> {foll_html}</div>'
            '</div>'
        )
        st.markdown(card_500, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("合計購入金額", f"{m500['total_cost']} 円（{len(df_500)}点）")
        c2.metric("合成的中率", f"{df_500['三連複的中率(%)'].sum():.2f}%")
        c3.metric("想定払戻レンジ", f"{df_500['想定払戻(円)'].min():,} 〜 {df_500['想定払戻(円)'].max():,} 円")
        st.dataframe(
            df_500[["券種", "組み合わせ", "馬名トリオ", "三連複オッズ", "三連複的中率(%)", "三連複期待値(EV)", "購入額(円)", "想定払戻(円)"]]
            .style.format({"三連複オッズ": "{:.1f}", "三連複的中率(%)": "{:.2f}%", "三連複期待値(EV)": "{:.2f}"}),
            width="stretch",
            hide_index=True,
        )

    elif current_mode == "300":
        m300 = budget_modes["mode_300"]
        df_300 = m300["tickets_df"]
        card_300 = (
            '<div class="bet-card">'
            f'<div class="bet-card-title">300円モード：期待値(EV)と的中率バランス厳選3点（計 {len(df_300)}点 × 100円 ＝ {m300["total_cost"]}円）</div>'
            f'<div style="font-size:0.86rem; color:#475569;">現在の狙い方スタイル（{style_esc}）に基づき、全券種から現実的かつ妙味のある3点を抽出しています。</div>'
            '</div>'
        )
        st.markdown(card_300, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("合計購入金額", f"{m300['total_cost']} 円（{len(df_300)}点）")
        c2.metric("平均期待値(EV)", f"{df_300['期待値(EV)'].mean():.2f}")
        c3.metric("最大想定払戻", f"{df_300['想定払戻(円)'].max():,} 円")
        st.dataframe(
            df_300[["券種", "組み合わせ", "対象馬名", "オッズ", "的中率(%)", "期待値(EV)", "購入額(円)", "想定払戻(円)"]]
            .style.format({"オッズ": "{:.1f}", "的中率(%)": "{:.2f}%", "期待値(EV)": "{:.2f}"}),
            width="stretch",
            hide_index=True,
        )

    else:
        m2000 = budget_modes["mode_2000"]
        df_trio_2000 = m2000["trio_df"]
        df_exacta_2000 = m2000["exacta_df"]
        df_wide_2000 = m2000["wide_df"]
        t_ax_html = render_horse_chips(m2000["trio_axis"], race_df, is_axis=True)
        t_fl_html = render_horse_chips(m2000["trio_followers"], race_df)
        ex_bx_html = render_horse_chips(m2000["exacta_box"], race_df)
        wd_bx_html = render_horse_chips(m2000["wide_box"], race_df)
        card_2000 = (
            '<div class="bet-card">'
            f'<div class="bet-card-title">2000円モード：3券種フルパッケージ（計 20点 × 100円 ＝ {m2000["total_cost"]:,}円）</div>'
            f'<div style="margin-bottom:5px;"><strong>1. 3連複 2頭軸5頭流し（500円）：</strong> 軸 {t_ax_html} ／ 相手 {t_fl_html}</div>'
            f'<div style="margin-bottom:5px;"><strong>2. 馬単 4頭ボックス（1200円）：</strong> {ex_bx_html}</div>'
            f'<div><strong>3. ワイド 3頭ボックス（300円）：</strong> {wd_bx_html}</div>'
            '</div>'
        )
        st.markdown(card_2000, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("合計投資額", f"{m2000['total_cost']:,} 円（20点）")
        c2.metric("3連複 5点合成的中率", f"{df_trio_2000['三連複的中率(%)'].sum():.2f}%")
        c3.metric("馬単 12点合成的中率", f"{df_exacta_2000['馬単的中率(%)'].sum():.2f}%")
        c4.metric("ワイド 3点延べ的中率", f"{df_wide_2000['ワイド的中率(%)'].sum():.2f}%")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 1. 3連複 2頭軸5頭流し（5点・500円）")
            st.dataframe(
                df_trio_2000[["組み合わせ", "馬名トリオ", "三連複オッズ", "三連複的中率(%)", "三連複期待値(EV)", "想定払戻(円)"]]
                .style.format({"三連複オッズ": "{:.1f}", "三連複的中率(%)": "{:.2f}%", "三連複期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
            st.markdown("#### 3. ワイド 3頭ボックス（3点・300円）")
            st.dataframe(
                df_wide_2000[["組み合わせ", "馬名ペア", "ワイドオッズ", "ワイド的中率(%)", "ワイド期待値(EV)", "想定払戻(円)"]]
                .style.format({"ワイドオッズ": "{:.1f}", "ワイド的中率(%)": "{:.2f}%", "ワイド期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
        with col_right:
            st.markdown("#### 2. 馬単 4頭ボックス（12点・1200円）")
            st.dataframe(
                df_exacta_2000[["組み合わせ", "馬名順列", "馬単オッズ", "馬単的中率(%)", "馬単期待値(EV)", "想定払戻(円)"]]
                .style.format({"馬単オッズ": "{:.1f}", "馬単的中率(%)": "{:.2f}%", "馬単期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )

    # ==========================================================================
    # 買い目保存 ＆ 即PAT連携アクションバー（200円モード完全対応）
    # ==========================================================================
    if current_mode == "200":
        proxy_modes = dict(budget_modes)
        proxy_modes["mode_300"] = budget_modes["mode_200"]
        active_ticket_rows = bet_tracker.extract_mode_ticket_rows("300", proxy_modes)
        for row_item in active_ticket_rows:
            if isinstance(row_item, dict) and "mode" in row_item:
                row_item["mode"] = "200円"
        ipat_data = bet_tracker.build_ipat_exports(race_meta, "300", proxy_modes, active_ticket_rows)
        ipat_data = {
            "human_text": str(ipat_data.get("human_text", "")).replace("300円", "200円"),
            "ipat_csv": str(ipat_data.get("ipat_csv", "")),
            "filename": str(ipat_data.get("filename", "ipat_200.csv")).replace("300", "200"),
        }
    else:
        active_ticket_rows = bet_tracker.extract_mode_ticket_rows(current_mode, budget_modes)
        ipat_data = bet_tracker.build_ipat_exports(race_meta, current_mode, budget_modes, active_ticket_rows)

    act_c1, act_c2, act_c3 = st.columns(3)
    with act_c1:
        if st.button(f"この{current_mode}円モード買い目を履歴に保存", type="primary", width="stretch"):
            saved_n, skipped_n = bet_tracker.append_bets_to_history(race_meta, active_ticket_rows)
            if saved_n > 0:
                st.success(f"{saved_n}点の買い目を保存しました。「戦績・回収率」タブから結果照合できます。")
            else:
                st.info(f"このレースの{current_mode}円モード買い目はすでに保存済みです（重複スキップ: {skipped_n}点）。")
    with act_c2:
        st.download_button(
            label="IPAT連携用CSVをダウンロード",
            data=ipat_data["ipat_csv"].encode("utf-8-sig"),
            file_name=ipat_data["filename"],
            mime="text/csv",
            width="stretch",
        )
    with act_c3:
        st.link_button(
            label="JRA 即PAT を開く",
            url="https://www.ipat.jra.go.jp/",
            width="stretch",
        )

    with st.expander("即PAT入力用テキスト・IPAT連携データを確認（右上のアイコンでコピー可能）"):
        st.caption("即PAT画面に入力しやすいまとめ表記です。右上のコピーボタンからメモやスマホへ貼り付けできます。")
        st.code(ipat_data["human_text"], language="text")
        st.caption("ipatgo 等のIPAT連動ツール向けCSVフォーマット:")
        st.code(ipat_data["ipat_csv"].strip(), language="text")

    # ==========================================================================
    # 出走表 ＋ 右側インライン詳細パネル
    # ==========================================================================
    valid_horse_nums = set(race_df["horse_num"].astype(int).tolist())
    sel_detail_h = st.session_state.get("selected_detail_horse")
    if sel_detail_h is not None and int(sel_detail_h) not in valid_horse_nums:
        sel_detail_h = None
        st.session_state["selected_detail_horse"] = None

    hdr_c1, hdr_c2 = st.columns([2.2, 1.4])
    with hdr_c1:
        st.markdown(
            f"### 出走表 <span style='font-size:0.86rem; color:#64748b; font-weight:600;'>{len(race_df)}頭（馬名クリックで右側に詳細・近5走を表示）</span>",
            unsafe_allow_html=True,
        )
    with hdr_c2:
        quick_opts = ["馬詳細パネルを開く（AI順位順）"] + [
            f"AI{int(r['ai_rank'])}位 | {int(r['horse_num']):02d} {r['horse_name']}"
            for _, r in race_df.iterrows()
        ]
        quick_idx = 0
        if sel_detail_h is not None:
            for q_i, (_, r_q) in enumerate(race_df.iterrows(), start=1):
                if int(r_q["horse_num"]) == int(sel_detail_h):
                    quick_idx = q_i
                    break

        quick_key = f"quick_horse_sel_{race_id_str}"
        expected_q = quick_opts[quick_idx]
        if st.session_state.get(quick_key) != expected_q:
            st.session_state[quick_key] = expected_q

        def _on_quick_horse_change(q_key: str, default_opt: str):
            val = st.session_state.get(q_key, default_opt)
            if val == default_opt:
                st.session_state["selected_detail_horse"] = None
            else:
                st.session_state["selected_detail_horse"] = int(val.split("|")[1].strip().split()[0])

        st.selectbox(
            "馬詳細クイック切替",
            quick_opts,
            key=quick_key,
            label_visibility="collapsed",
            on_change=_on_quick_horse_change,
            args=(quick_key, quick_opts[0]),
        )

    table_rows_html = []
    for _, r in race_df.iterrows():
        ai_rank = int(r["ai_rank"])
        waku = int(np.clip(r["frame_num"], 1, 8))
        umaban = int(r["horse_num"])
        is_sel_row = (sel_detail_h is not None and int(sel_detail_h) == umaban)

        base_row_cls = "row-rank-1" if ai_rank == 1 else ("row-rank-2" if ai_rank == 2 else ("row-rank-3" if ai_rank == 3 else ""))
        row_cls = f"{base_row_cls} row-selected".strip() if is_sel_row else base_row_cls
        umaban_cls = "umaban-txt umaban-top" if ai_rank <= 2 else "umaban-txt"

        ability = float(r["ability_score"])
        bar_pct = int(np.clip((ability - 35.0) / (75.0 - 35.0) * 100.0, 8, 100))
        if ability >= 56.0:
            bar_color = "#dc2626"
        elif ability >= 50.0:
            bar_color = "#f97316"
        else:
            bar_color = "#2563eb"

        prob_cls = "prob-hot" if ai_rank == 1 else ("prob-warm" if ai_rank == 2 else "prob-norm")
        ev_val = float(r["単勝期待値(EV)"])
        ev_style = "color:#dc2626; font-weight:800;" if ev_val >= ev_threshold else "color:#475569; font-weight:600;"

        detail_qs = urlencode({
            "horse": str(umaban),
            "race_id": race_id_str,
            "date": current_date_str,
            "mode": current_mode,
            "style": selected_style_mode,
            "ai_w": f"{ai_weight:.2f}",
            "ev_th": f"{ev_threshold:.2f}",
            "ex1": "1" if exclude_fav1 else "0",
        })
        detail_href = f"?{detail_qs}"
        mark_esc = html.escape(str(r["印"]))
        hname_esc = html.escape(str(r["horse_name"]))
        jock_esc = html.escape(str(r["jockey_id"]))

        row_str = (
            f'<tr class="{row_cls}">'
            f'<td><div class="pop-tag">{int(r["popularity"])}人気</div><div class="odds-val">{float(r["win_odds"]):.1f}</div></td>'
            f'<td style="font-weight:800;">{ai_rank}</td>'
            f'<td><span class="waku-badge waku-{waku}">{waku}</span></td>'
            f'<td><span class="{umaban_cls}">{umaban}</span></td>'
            f'<td><span class="mark-badge">{mark_esc}</span></td>'
            f'<td class="horse-cell">'
            f'<div class="horse-title"><a class="horse-link" href="{detail_href}" target="_self">{hname_esc}</a></div>'
            f'<div class="jockey-sub">{jock_esc} &nbsp; {float(r["weight_carried"]):.1f}</div></td>'
            f'<td>{r["signal_html"]}</td>'
            f'<td>{r["conf_html"]}</td>'
            f'<td><div class="ability-wrap"><span class="ability-num">{ability:.1f}</span>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div></div></div></td>'
            f'<td class="{prob_cls}">{float(r["pred_win_pct"]):.1f}%</td>'
            f'<td class="{prob_cls}">{float(r["pred_quinella_pct"]):.1f}%</td>'
            f'<td class="{prob_cls}">{float(r["pred_place_pct"]):.1f}%</td>'
            f'<td style="{ev_style}">{ev_val:.2f}</td>'
            f'</tr>'
        )
        table_rows_html.append(row_str)

    tbody_str = "".join(table_rows_html)
    shutuba_html = (
        '<div class="shutuba-wrapper"><table class="shutuba-table">'
        '<thead><tr>'
        '<th>単勝オッズ ▼</th><th>AI順位 ▼</th><th>枠</th><th>馬番</th><th>印</th>'
        '<th style="text-align:left; padding-left:10px;">馬名 / 騎手 ▼</th>'
        '<th>シグナル</th><th>信頼度</th><th>能力値 ▼</th>'
        '<th>勝率 ▼</th><th>2着内率 ▼</th><th>3着内率 ▼</th><th>単勝EV ▼</th>'
        '</tr></thead>'
        f'<tbody>{tbody_str}</tbody>'
        '</table></div>'
    )

    if sel_detail_h is not None:
        col_tbl, col_panel = st.columns([1.62, 1.0], gap="medium")
        with col_tbl:
            st.markdown(shutuba_html, unsafe_allow_html=True)
        with col_panel:
            horse_detail_view.render_horse_side_panel(
                race_df=race_df,
                selected_horse_num=int(sel_detail_h),
                race_id=race_id_str,
                date_str=str(r_info["date"]),
                ev_threshold=ev_threshold,
                exclude_fav1=exclude_fav1,
                model=model,
                feature_cols=feature_cols,
                category_maps=category_maps,
                train_model=train_model,
            )
    else:
        st.markdown(shutuba_html, unsafe_allow_html=True)

    # ==========================================================================
    # 券種別 詳細データ一覧タブ
    # ==========================================================================
    def allocate_budget(df_target: pd.DataFrame, ev_col: str, odds_col: str, total_budget: int) -> pd.DataFrame:
        res = df_target.copy()
        n_rows = len(res)
        if n_rows == 0:
            return res
        weights = res[ev_col] / res[ev_col].sum()
        raw_alloc = (total_budget * weights) // 100 * 100
        if total_budget >= n_rows * 100:
            raw_alloc = raw_alloc.clip(lower=100)
        res["推奨購入額(円)"] = raw_alloc.astype(int)
        res["想定払戻(円)"] = (res["推奨購入額(円)"] * res[odds_col]).astype(int)
        return res

    def format_horse_list(nums: list[int], df_horses: pd.DataFrame) -> str:
        num_to_name = dict(zip(df_horses["horse_num"].astype(int), df_horses["horse_name"]))
        return " , ".join([f"{n:02d}({num_to_name.get(n, '')})" for n in nums])

    st.markdown("### 券種別 詳細データ一覧")
    tab_win, tab_umaren, tab_wide, tab_trio = st.tabs([
        "単勝 一覧",
        "馬連 一覧",
        "ワイド 一覧",
        "三連複 フォーメーション・単品",
    ])

    rename_map = {
        "horse_num": "馬番",
        "horse_name": "馬名",
        "win_odds": "単勝オッズ",
        "pred_win_pct": "予測勝率(%)",
    }

    with tab_win:
        win_buy = race_df[(race_df["単勝期待値(EV)"] >= ev_threshold) & (race_df["pred_win_pct"] >= min_win_prob)].copy()
        if len(win_buy) > 0:
            win_buy = allocate_budget(win_buy, "単勝期待値(EV)", "win_odds", budget)
            st.dataframe(
                win_buy[["印", "horse_num", "horse_name", "win_odds", "pred_win_pct", "単勝期待値(EV)", "推奨購入額(円)", "想定払戻(円)"]]
                .rename(columns=rename_map)
                .style.format({"単勝オッズ": "{:.1f}", "予測勝率(%)": "{:.1f}%", "単勝期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("条件を満たす単勝買い目がありません。")

    with tab_umaren:
        umaren_buy = (
            pairs_df[(pairs_df["馬連期待値(EV)"] >= ev_threshold) & (pairs_df["馬連的中率(%)"] >= min_pair_prob)]
            .sort_values("馬連期待値(EV)", ascending=False)
            .head(max_tickets)
        )
        if len(umaren_buy) > 0:
            umaren_buy = allocate_budget(umaren_buy, "馬連期待値(EV)", "馬連オッズ", budget)
            st.dataframe(
                umaren_buy[["組み合わせ", "馬名ペア", "馬連オッズ", "馬連的中率(%)", "馬連期待値(EV)", "推奨購入額(円)", "想定払戻(円)"]]
                .style.format({"馬連オッズ": "{:.1f}", "馬連的中率(%)": "{:.2f}%", "馬連期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("条件を満たす馬連買い目がありません。")

    with tab_wide:
        wide_buy = (
            pairs_df[(pairs_df["ワイド期待値(EV)"] >= ev_threshold) & (pairs_df["ワイド的中率(%)"] >= min_pair_prob)]
            .sort_values("ワイド期待値(EV)", ascending=False)
            .head(max_tickets)
        )
        if len(wide_buy) > 0:
            wide_buy = allocate_budget(wide_buy, "ワイド期待値(EV)", "ワイドオッズ", budget)
            st.dataframe(
                wide_buy[["組み合わせ", "馬名ペア", "ワイドオッズ", "ワイド的中率(%)", "ワイド期待値(EV)", "推奨購入額(円)", "想定払戻(円)"]]
                .style.format({"ワイドオッズ": "{:.1f}", "ワイド的中率(%)": "{:.2f}%", "ワイド期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("条件を満たすワイド買い目がありません。")

    with tab_trio:
        form = trio_strategies["formation"]
        form_df = form["tickets_df"]
        if len(form_df) > 0:
            form_df = allocate_budget(form_df, "三連複期待値(EV)", "三連複オッズ", budget)
            st.markdown(
                f"**自動フォーメーション（全 {len(form_df)} 点）** — "
                f"1列目: `{format_horse_list(form['col1'], race_df)}` ／ "
                f"2列目: `{format_horse_list(form['col2'], race_df)}` ／ "
                f"3列目: `{format_horse_list(form['col3'], race_df)}`"
            )
            st.dataframe(
                form_df[["組み合わせ", "馬名トリオ", "三連複オッズ", "三連複的中率(%)", "三連複期待値(EV)", "推奨購入額(円)", "想定払戻(円)"]]
                .style.format({"三連複オッズ": "{:.1f}", "三連複的中率(%)": "{:.2f}%", "三連複期待値(EV)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("フォーメーションを構築できる候補が不足しています。")

# ==============================================================================
# タブ2：WIN5 期待値予想ツール
# ==============================================================================
with tab_win5:
    win5_engine.render_win5_tab(
        model=model,
        feature_cols=feature_cols,
        category_maps=category_maps,
        train_model=train_model,
        shutuba_scraper=shutuba_scraper,
        ai_weight=ai_weight,
        selected_style_mode=selected_style_mode,
        exclude_fav1=exclude_fav1,
        style_short_badge=style_short_badge,
    )

# ==============================================================================
# タブ3：戦績・回収率ダッシュボード
# ==============================================================================
with tab_stats:
    st.markdown("### 予想買い目の戦績・回収率管理")

    hist_df = bet_tracker.load_bet_history()

    top_a1, top_a2 = st.columns([2, 1])
    with top_a1:
        if st.button("未確定レースの結果をWebから自動照合して回収率を更新", type="primary", width="stretch"):
            with st.spinner("確定結果と公式払戻金を照合中..."):
                summary = bet_tracker.settle_pending_bets()
                if summary["settled_tickets"] > 0:
                    st.success(
                        f"照合完了: {summary['checked_races']}レース / {summary['settled_tickets']}点を判定しました（うち的中: {summary['hit_tickets']}点）。"
                    )
                    st.rerun()
                else:
                    st.info("新しく確定できた未照合レースはありませんでした（発走前、またはすべて照合済みです）。")
    with top_a2:
        if not hist_df.empty:
            st.download_button(
                label="戦績履歴CSVをダウンロード",
                data=hist_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="kazuyan_ai_bet_history.csv",
                mime="text/csv",
                width="stretch",
            )

    if hist_df.empty:
        st.info(
            "保存されている買い目がまだありません。「レース予想・買い目」タブで200円 / 300円 / 500円 / 2000円モードを選び、「この買い目を履歴に保存」を押すとここに蓄積されます。"
        )
    else:
        settled_df = hist_df[hist_df["status"].isin(["的中", "不的中"])].copy()
        pending_count = int((hist_df["status"] == "未確定").sum())

        total_bet_settled = int(settled_df["amount"].sum()) if not settled_df.empty else 0
        total_payout = int(settled_df["payout"].sum()) if not settled_df.empty else 0
        net_profit = total_payout - total_bet_settled
        roi_pct = (total_payout / total_bet_settled * 100.0) if total_bet_settled > 0 else 0.0

        if not settled_df.empty:
            race_grp = settled_df.groupby(["race_id", "mode"]).agg(
                投資額=("amount", "sum"),
                払戻額=("payout", "sum"),
                的中点数=("status", lambda s: int((s == "的中").sum())),
            )
            race_hit_rate = (race_grp["的中点数"] > 0).mean() * 100.0
            hit_races = int((race_grp["的中点数"] > 0).sum())
            total_races = len(race_grp)
        else:
            race_hit_rate = 0.0
            hit_races = 0
            total_races = 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("確定済み総投資額", f"{total_bet_settled:,} 円", f"未確定: {pending_count}点")
        k2.metric("総払戻額", f"{total_payout:,} 円")
        k3.metric("トータル収支", f"{net_profit:+,} 円")
        k4.metric("総合回収率", f"{roi_pct:.1f}%")
        k5.metric("レース的中率", f"{race_hit_rate:.1f}%", f"{hit_races} / {total_races} 戦")

        if not settled_df.empty:
            def summarize_by_col(df_in: pd.DataFrame, group_col: str, label_name: str) -> pd.DataFrame:
                g = df_in.groupby(group_col, as_index=False).agg(
                    購入点数=("amount", "count"),
                    的中数=("status", lambda s: int((s == "的中").sum())),
                    投資額=("amount", "sum"),
                    払戻額=("payout", "sum"),
                )
                g["収支"] = g["払戻額"] - g["投資額"]
                g["的中率(%)"] = (g["的中数"] / g["購入点数"]) * 100.0
                g["回収率(%)"] = (g["払戻額"] / g["投資額"]) * 100.0
                return g.rename(columns={group_col: label_name})

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### モード別 回収率（200円 / 300円 / 500円 / 2000円）")
                mode_summary = summarize_by_col(settled_df, "mode", "モード")
                st.dataframe(
                    mode_summary.style.format({
                        "投資額": "{:,}",
                        "払戻額": "{:,}",
                        "収支": "{:+,}",
                        "的中率(%)": "{:.1f}%",
                        "回収率(%)": "{:.1f}%",
                    }),
                    width="stretch",
                    hide_index=True,
                )

            with col_s2:
                st.markdown("#### 券種別 回収率")
                type_summary = summarize_by_col(settled_df, "ticket_type", "券種")
                st.dataframe(
                    type_summary.style.format({
                        "投資額": "{:,}",
                        "払戻額": "{:,}",
                        "収支": "{:+,}",
                        "的中率(%)": "{:.1f}%",
                        "回収率(%)": "{:.1f}%",
                    }),
                    width="stretch",
                    hide_index=True,
                )

        st.markdown("#### レース別 戦績一覧")
        race_summary = (
            hist_df.groupby(["date", "venue", "race_num", "race_title", "race_id", "mode"], as_index=False)
            .agg(
                点数=("amount", "count"),
                投資額=("amount", "sum"),
                払戻額=("payout", "sum"),
                状態=("status", lambda s: "未確定" if (s == "未確定").any() else ("的中" if (s == "的中").any() else "不的中")),
                確定結果=("actual_result", "first"),
            )
            .sort_values(["date", "race_id"], ascending=[False, False])
        )
        race_summary["収支"] = race_summary.apply(
            lambda r: 0 if r["状態"] == "未確定" else int(r["払戻額"] - r["投資額"]),
            axis=1,
        )
        race_summary["回収率(%)"] = race_summary.apply(
            lambda r: 0.0 if r["状態"] == "未確定" or r["投資額"] == 0 else (r["払戻額"] / r["投資額"]) * 100.0,
            axis=1,
        )
        st.dataframe(
            race_summary[["date", "venue", "race_num", "race_title", "mode", "点数", "投資額", "払戻額", "収支", "回収率(%)", "状態", "確定結果"]]
            .rename(columns={"date": "開催日", "venue": "場", "race_num": "R", "race_title": "レース名", "mode": "モード"})
            .style.format({"投資額": "{:,}", "払戻額": "{:,}", "収支": "{:+,}", "回収率(%)": "{:.1f}%"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### 買い目明細ログ")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            filter_status = st.multiselect("ステータスで絞り込み", ["未確定", "的中", "不的中"], default=["未確定", "的中", "不的中"])
        with f_col2:
            filter_mode = st.multiselect("モードで絞り込み", ["200円", "300円", "500円", "2000円"], default=["200円", "300円", "500円", "2000円"])

        view_df = hist_df[hist_df["status"].isin(filter_status) & hist_df["mode"].isin(filter_mode)].copy()
        view_df["combination"] = view_df["combination"].str.replace(">", "→", regex=False)
        st.dataframe(
            view_df[["date", "venue", "race_num", "mode", "ticket_type", "combination", "horse_names", "amount", "pred_odds", "status", "payout", "actual_result"]]
            .rename(columns={
                "date": "開催日",
                "venue": "場",
                "race_num": "R",
                "mode": "モード",
                "ticket_type": "券種",
                "combination": "組み合わせ",
                "horse_names": "対象馬名",
                "amount": "投資(円)",
                "pred_odds": "予想時オッズ",
                "status": "判定",
                "payout": "払戻(円)",
                "actual_result": "確定上位",
            })
            .style.format({"予想時オッズ": "{:.1f}", "投資(円)": "{:,}", "払戻(円)": "{:,}"}),
            width="stretch",
            hide_index=True,
        )

        with st.expander("履歴の削除・リセット"):
            del_options = [
                f"{r['date']} {r['venue']}{r['race_num']}R ({r['mode']}) | {r['race_id']}"
                for _, r in race_summary.iterrows()
            ]
            if del_options:
                sel_del = st.selectbox("削除するレース履歴を選択", del_options)
                if st.button("選択したレースの履歴を削除"):
                    target_rid = sel_del.split("|")[-1].strip()
                    target_mode = sel_del.split("(")[-1].split(")")[0].strip()
                    bet_tracker.delete_race_from_history(target_rid, target_mode)
                    st.success("削除しました。")
                    st.rerun()

            st.divider()
            confirm_clear = st.checkbox("すべての戦績履歴を完全に削除する（元に戻せません）")
            if st.button("全履歴をリセット", disabled=not confirm_clear):
                bet_tracker.clear_all_history()
                st.success("全履歴をリセットしました。")
                st.rerun()

# ==============================================================================
# タブ4：ヘルプ・算出ロジック解説画面
# ==============================================================================
with tab_help:
    t_badge_esc = html.escape(train_status["badge"])
    t_detail_esc = html.escape(train_status["detail"])
    help_html_1 = (
        '<div class="help-card">'
        '<h4>1. KAZUYAN AI モデルと学習データの状態</h4>'
        f'<p style="margin-bottom:8px;"><strong>ステータス：</strong> {t_badge_esc}</p>'
        f'<p style="color:#475569; margin-bottom:10px;">{t_detail_esc}</p>'
        '<div class="formula-box">'
        'JRA実データ5年分を収集して再学習するコマンド手順:<br>'
        '1. python 05_scrape_history.py --start-date 2022-01-01 --end-date 2026-11-24 --min-race-num 1<br>'
        '2. python 02_train_model.py'
        '</div>'
        '</div>'
        '<div class="help-card">'
        '<h4>2. 馬詳細サイドパネル＆ボタン操作の使い方</h4>'
        '<table class="help-table">'
        '<thead><tr><th>機能・ボタン</th><th>動作内容</th></tr></thead>'
        '<tbody>'
        '<tr><td><strong>出走表の馬名クリック</strong></td><td>別ウィンドウを開かず、出走表の右側に馬詳細パネルが表示されます。シグナル判定理由・AI特徴量寄与度・基本情報（性齢/厩舎/血統/間隔/通算）・近5走成績を一覧でき、「‹ 前の馬 / 次の馬 ›」で全頭を素早く確認できます。</td></tr>'
        '<tr><td><strong>開催馬場ボタン（東京・京都 等）</strong></td><td>開催日のレース一覧を取得すると、当日の開催場が丸角ボタンで並びます。タップするだけで別会場の同R（またはメインR）へ即座に切り替わります。</td></tr>'
        '<tr><td><strong>← 前R / 1R〜12R / 次R →</strong></td><td>選択中の競馬場の前後のレースや指定R番号へワンタップで移動し、出馬表と全オッズを自動取得します。</td></tr>'
        '<tr><td><strong>最新更新</strong></td><td>発走直前のオッズ変動を取り込むため、現在表示しているレースのオッズを再取得します。</td></tr>'
        '</tbody>'
        '</table>'
        '</div>'
        '<div class="help-card">'
        '<h4>3. 狙い方スタイルとAI独自度の仕組み</h4>'
        '<table class="help-table">'
        '<thead><tr><th>設定項目</th><th>内容と効果</th></tr></thead>'
        '<tbody>'
        '<tr><td><strong>標準バランス（初期設定）</strong></td><td>三連複の「1頭目の軸」をAI予測上位4頭以内の実力馬から選び、「2頭目の軸と相手5頭」に期待値の高い中穴を絡める王道スタイルです。</td></tr>'
        '<tr><td><strong>堅実重視（的中率優先）</strong></td><td>軸2頭・BOX候補ともにAI実力上位馬に限定し、的中率の重みを強めて現実的な決着を狙います。</td></tr>'
        '<tr><td><strong>穴狙い特化（高配当EV優先）</strong></td><td>的中率よりも期待値（EV）の大きさを優先して高配当を狙います。</td></tr>'
        '<tr><td><strong>AI独自度（初期値 75%）</strong></td><td>AI純粋確率（75%）と単勝オッズから逆算した市場評価（25%）を合成します。</td></tr>'
        '</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(help_html_1, unsafe_allow_html=True)