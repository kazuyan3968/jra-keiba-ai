import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

try:
    import lightgbm as lgb
except Exception:
    lgb = None

CATEGORICAL_COLS = ["venue", "surface", "track_condition"]

FEATURE_COLS = [
    "venue",
    "surface",
    "distance",
    "track_condition",
    "frame_num",
    "horse_num",
    "field_size",
    "frame_ratio",
    "weight_carried",
    "horse_weight",
    "weight_diff",
    "abs_weight_diff",
    "weight_ratio",
    "jockey_win_rate",
    "jockey_place_rate",
    "jockey_surface_win_rate",
    "jockey_venue_win_rate",
    "rel_jockey_win_rate",
    "jockey_upgrade",
    "trainer_win_rate",
    "trainer_place_rate",
    "prev1_rank",
    "prev2_rank",
    "prev3_rank",
    "avg_prev_rank",
    "best_prev_rank",
    "rank_trend",
    "rel_prev_rank",
    "prev1_time_diff",
    "rel_time_diff",
]


class PureLGBMBooster:
    """
    lightgbm パッケージが無い環境（Streamlit Cloud等）でも、
    model_to_string() のテキストから LightGBM の二分木推論を再現する純粋 NumPy エンジン。
    """

    def __init__(self, model_str: str):
        self.trees = []
        self.sigmoid = 1.0
        self._parse_model_str(model_str)

    def _parse_model_str(self, model_str: str):
        lines = model_str.splitlines()
        cur_tree = None
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("objective="):
                m_sig = re.search(r"sigmoid:([0-9eE.+-]+)", line)
                if m_sig:
                    self.sigmoid = float(m_sig.group(1))
            elif line.startswith("Tree="):
                if cur_tree is not None:
                    self.trees.append(self._finalize_tree(cur_tree))
                cur_tree = {}
            elif line == "end of trees":
                if cur_tree is not None:
                    self.trees.append(self._finalize_tree(cur_tree))
                    cur_tree = None
            elif cur_tree is not None and "=" in line:
                k, v = line.split("=", 1)
                cur_tree[k] = v.strip()

        if cur_tree is not None:
            self.trees.append(self._finalize_tree(cur_tree))

    @staticmethod
    def _finalize_tree(t: dict) -> dict:
        num_leaves = int(t.get("num_leaves", 1))
        leaf_value = np.fromstring(t.get("leaf_value", "0.0"), sep=" ", dtype=np.float64)
        if num_leaves <= 1 or "split_feature" not in t or not t["split_feature"]:
            return {"num_leaves": 1, "leaf_value": leaf_value}

        split_feature = np.fromstring(t["split_feature"], sep=" ", dtype=np.int32)
        threshold = np.fromstring(t["threshold"], sep=" ", dtype=np.float64)
        decision_type = np.fromstring(t["decision_type"], sep=" ", dtype=np.int32)
        left_child = np.fromstring(t["left_child"], sep=" ", dtype=np.int32)
        right_child = np.fromstring(t["right_child"], sep=" ", dtype=np.int32)
        cat_boundaries = (
            np.fromstring(t["cat_boundaries"], sep=" ", dtype=np.int32)
            if "cat_boundaries" in t and t["cat_boundaries"]
            else np.array([], dtype=np.int32)
        )
        cat_threshold = (
            np.fromstring(t["cat_threshold"], sep=" ", dtype=np.uint32)
            if "cat_threshold" in t and t["cat_threshold"]
            else np.array([], dtype=np.uint32)
        )
        return {
            "num_leaves": num_leaves,
            "split_feature": split_feature,
            "threshold": threshold,
            "decision_type": decision_type,
            "left_child": left_child,
            "right_child": right_child,
            "leaf_value": leaf_value,
            "cat_boundaries": cat_boundaries,
            "cat_threshold": cat_threshold,
        }

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            cols_arr = []
            cat_flags = []
            for col in X.columns:
                s = X[col]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    cols_arr.append(s.cat.codes.to_numpy(dtype=np.float64))
                    cat_flags.append(True)
                else:
                    cols_arr.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64))
                    cat_flags.append(False)
            X_mat = np.column_stack(cols_arr) if cols_arr else np.zeros((len(X), 0), dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)
            cat_flags = [False] * X_mat.shape[1]

        n_samples = X_mat.shape[0]
        raw_scores = np.zeros(n_samples, dtype=np.float64)

        for tree in self.trees:
            if tree["num_leaves"] <= 1:
                raw_scores += float(tree["leaf_value"][0])
                continue

            sf = tree["split_feature"]
            th = tree["threshold"]
            dt = tree["decision_type"]
            lc = tree["left_child"]
            rc = tree["right_child"]
            lv = tree["leaf_value"]
            cb = tree["cat_boundaries"]
            ct = tree["cat_threshold"]

            for i in range(n_samples):
                node = 0
                while node >= 0:
                    f_idx = int(sf[node])
                    val = X_mat[i, f_idx]
                    d_type = int(dt[node])
                    is_cat = (d_type & 1) == 1
                    default_left = (d_type & 2) != 0
                    missing_type = (d_type >> 2) & 3

                    if is_cat or cat_flags[f_idx]:
                        c_code = -1 if np.isnan(val) else int(val)
                        if missing_type == 0 and c_code < 0:
                            c_code = 0
                        is_missing = (c_code < 0) or (missing_type == 1 and c_code == 0)
                        if is_missing:
                            node = int(lc[node]) if default_left else int(rc[node])
                        else:
                            cat_idx = int(round(th[node]))
                            st_i = int(cb[cat_idx])
                            ed_i = int(cb[cat_idx + 1])
                            w_i = c_code >> 5
                            b_i = c_code & 31
                            in_set = (0 <= w_i < (ed_i - st_i)) and ((int(ct[st_i + w_i]) & (1 << b_i)) != 0)
                            node = int(lc[node]) if in_set else int(rc[node])
                    else:
                        if missing_type == 0:
                            v_num = 0.0 if np.isnan(val) else val
                            is_missing = False
                        elif missing_type == 1:
                            is_missing = np.isnan(val) or (val == 0.0)
                            v_num = val
                        else:
                            is_missing = np.isnan(val)
                            v_num = val

                        if is_missing:
                            node = int(lc[node]) if default_left else int(rc[node])
                        else:
                            node = int(lc[node]) if v_num <= th[node] else int(rc[node])

                raw_scores[i] += float(lv[~node])

        return 1.0 / (1.0 + np.exp(-self.sigmoid * raw_scores))


class BoosterBinaryWrapper:
    """
    lightgbm.Booster テキストから predict_proba を提供するラッパー。
    lightgbm が導入済みの環境ではネイティブ Booster を、未導入環境では PureLGBMBooster を自動選択する。
    """

    def __init__(self, booster_str: str):
        self.booster_str = str(booster_str)
        if lgb is not None:
            self._booster = lgb.Booster(model_str=self.booster_str)
        else:
            self._booster = PureLGBMBooster(self.booster_str)

    def predict_proba(self, X) -> np.ndarray:
        p1 = np.asarray(self._booster.predict(X), dtype=float).reshape(-1)
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        return np.column_stack([1.0 - p1, p1])


def load_model_artifact(path: str = "models/keiba_lgbm.pkl") -> dict:
    """
    保存済みモデルを読み込み、booster_str 形式の場合は自動で BoosterBinaryWrapper を復元する。
    """
    raw_obj = joblib.load(path)
    if not isinstance(raw_obj, dict):
        raise ValueError("モデルファイル形式が不正です。")
    art = dict(raw_obj)
    if "booster_str" in art and art["booster_str"]:
        art["model"] = BoosterBinaryWrapper(art["booster_str"])
    return art


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    historical_races.csv や出馬表DataFrameの列名の揺れ（rank / finish_rank / 着順 / target など）を
    標準列名へ自動統一する。
    """
    data = df.copy()
    alias_groups = {
        "finish_rank": ["finish_rank", "rank", "result_rank", "order_of_finish", "finish_order", "着順"],
        "win_odds": ["win_odds", "odds", "単勝オッズ", "単勝"],
        "horse_num": ["horse_num", "umaban", "馬番"],
        "frame_num": ["frame_num", "wakuban", "枠番", "枠"],
        "weight_carried": ["weight_carried", "kinryo", "斤量"],
        "horse_weight": ["horse_weight", "weight", "馬体重"],
        "weight_diff": ["weight_diff", "増減", "馬体重増減"],
        "jockey_id": ["jockey_id", "jockey", "jockey_name", "騎手"],
        "trainer_id": ["trainer_id", "trainer", "trainer_name", "調教師", "厩舎"],
        "venue": ["venue", "place", "kaisai", "競馬場", "開催"],
        "surface": ["surface", "track_type", "course_type", "コース", "トラック"],
        "distance": ["distance", "dist", "距離"],
        "track_condition": ["track_condition", "condition", "baba", "馬場", "馬場状態"],
        "date": ["date", "race_date", "開催日", "日付"],
        "race_id": ["race_id", "id", "レースID"],
        "horse_name": ["horse_name", "name", "馬名"],
    }

    for std_col, candidates in alias_groups.items():
        if std_col in data.columns:
            continue
        for cand in candidates:
            if cand in data.columns:
                data[std_col] = data[cand]
                break

    if "finish_rank" not in data.columns:
        for bin_col in ("target", "is_win", "win"):
            if bin_col in data.columns:
                b_vals = pd.to_numeric(data[bin_col], errors="coerce").fillna(0)
                data["finish_rank"] = np.where(b_vals == 1, 1, 7)
                break

    return data


def _clean_person_name(s: pd.Series) -> pd.Series:
    cleaned = s.fillna("UNKNOWN").astype(str).str.strip()
    cleaned = cleaned.str.replace(r"^[▲△☆◇★\s]+", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    cleaned = cleaned.where(cleaned != "", "UNKNOWN")
    return cleaned


def _compute_expanding_bayes_rate(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    prior_mean: float,
    prior_weight: float,
) -> np.ndarray:
    """
    時系列リークを完全に防ぐため、開催日(date)単位で「前日までの累積実績」のみから
    ベイズ平滑化勝率／複勝率を算出する。
    """
    day_grp = (
        df.groupby(["date"] + group_cols, as_index=False)
        .agg(_cnt=(target_col, "count"), _sum=(target_col, "sum"))
        .sort_values("date")
    )
    day_grp["_cum_cnt"] = day_grp.groupby(group_cols)["_cnt"].cumsum() - day_grp["_cnt"]
    day_grp["_cum_sum"] = day_grp.groupby(group_cols)["_sum"].cumsum() - day_grp["_sum"]
    day_grp["_rate"] = (day_grp["_cum_sum"] + prior_weight * prior_mean) / (day_grp["_cum_cnt"] + prior_weight)
    merged = df[["date"] + group_cols].merge(
        day_grp[["date"] + group_cols + ["_rate"]],
        on=["date"] + group_cols,
        how="left",
    )
    return merged["_rate"].fillna(prior_mean).to_numpy(dtype=float)


def build_entity_stats_artifact(train_df: pd.DataFrame) -> dict:
    """
    学習完了時点の過去データ全体から、推論時に参照する騎手・厩舎・コース相性の
    ベイズ平滑化実績辞書を構築する。
    """
    df = normalize_raw_columns(train_df)
    df["jockey_clean"] = _clean_person_name(df["jockey_id"])
    df["trainer_clean"] = _clean_person_name(df["trainer_id"])
    df["surface_clean"] = df["surface"].fillna("芝").astype(str).str.strip()
    df["venue_clean"] = df["venue"].fillna("東京").astype(str).str.strip()
    df["is_win"] = (pd.to_numeric(df["finish_rank"], errors="coerce") == 1).astype(float)
    df["is_place"] = (pd.to_numeric(df["finish_rank"], errors="coerce") <= 3).astype(float)

    base_win = float(df["is_win"].mean()) if len(df) > 0 else 0.075
    base_place = float(df["is_place"].mean()) if len(df) > 0 else 0.225

    def _agg_map(cols: list[str], target: str, prior_m: float, prior_w: float) -> dict:
        g = df.groupby(cols).agg(cnt=(target, "count"), s=(target, "sum"))
        rates = (g["s"] + prior_w * prior_m) / (g["cnt"] + prior_w)
        if len(cols) == 1:
            return {str(k): round(float(v), 5) for k, v in rates.items()}
        return {"|".join(str(x) for x in k): round(float(v), 5) for k, v in rates.items()}

    return {
        "base_win": round(float(base_win), 5),
        "base_place": round(float(base_place), 5),
        "jockey_win": _agg_map(["jockey_clean"], "is_win", base_win, 35.0),
        "jockey_place": _agg_map(["jockey_clean"], "is_place", base_place, 35.0),
        "jockey_surface_win": _agg_map(["jockey_clean", "surface_clean"], "is_win", base_win, 25.0),
        "jockey_venue_win": _agg_map(["jockey_clean", "venue_clean"], "is_win", base_win, 20.0),
        "trainer_win": _agg_map(["trainer_clean"], "is_win", base_win, 35.0),
        "trainer_place": _agg_map(["trainer_clean"], "is_place", base_place, 35.0),
    }


def create_features(
    df: pd.DataFrame,
    category_maps: dict | None = None,
    is_training: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    出馬表・過去走データからLightGBM用の特徴量DataFrameとメタ辞書を生成する。
    """
    data = normalize_raw_columns(df)
    if "race_id" not in data.columns:
        data["race_id"] = "SINGLE_RACE"
    data["race_id"] = data["race_id"].astype(str)
    if "date" not in data.columns:
        data["date"] = "2026-01-01"
    data["date"] = data["date"].astype(str)

    num_defaults = {
        "distance": 1800.0,
        "frame_num": 4.0,
        "horse_num": 8.0,
        "weight_carried": 56.0,
        "horse_weight": 476.0,
        "weight_diff": 0.0,
        "prev1_rank": 7.0,
        "prev2_rank": 7.0,
        "prev3_rank": 7.0,
        "prev1_time_diff": 0.8,
    }
    for col, default_val in num_defaults.items():
        if col not in data.columns:
            data[col] = default_val
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(default_val)

    for col in CATEGORICAL_COLS:
        if col not in data.columns:
            data[col] = "UNKNOWN"
        data[col] = data[col].fillna("UNKNOWN").astype(str).str.strip()
        data.loc[data[col] == "", col] = "UNKNOWN"

    if "jockey_id" not in data.columns:
        data["jockey_id"] = "UNKNOWN"
    if "trainer_id" not in data.columns:
        data["trainer_id"] = "UNKNOWN"

    data["jockey_clean"] = _clean_person_name(data["jockey_id"])
    data["trainer_clean"] = _clean_person_name(data["trainer_id"])

    if "prev1_jockey_id" in data.columns:
        data["prev1_jockey_clean"] = _clean_person_name(data["prev1_jockey_id"])
    else:
        data["prev1_jockey_clean"] = "UNKNOWN"

    data["field_size"] = data.groupby("race_id")["horse_num"].transform("count").clip(lower=1).astype(float)
    data["frame_ratio"] = data["horse_num"] / data["field_size"]
    data["abs_weight_diff"] = data["weight_diff"].abs()
    data["weight_ratio"] = data["weight_carried"] / data["horse_weight"].clip(lower=350.0, upper=650.0)

    data["avg_prev_rank"] = (data["prev1_rank"] + data["prev2_rank"] + data["prev3_rank"]) / 3.0
    data["best_prev_rank"] = np.minimum(np.minimum(data["prev1_rank"], data["prev2_rank"]), data["prev3_rank"])
    data["rank_trend"] = data["prev3_rank"] - data["prev1_rank"]

    race_avg_rank = data.groupby("race_id")["avg_prev_rank"].transform("mean")
    data["rel_prev_rank"] = data["avg_prev_rank"] - race_avg_rank

    race_avg_tdiff = data.groupby("race_id")["prev1_time_diff"].transform("mean")
    data["rel_time_diff"] = data["prev1_time_diff"] - race_avg_tdiff

    if is_training and "finish_rank" in data.columns:
        data["_is_win"] = (pd.to_numeric(data["finish_rank"], errors="coerce") == 1).astype(float)
        data["_is_place"] = (pd.to_numeric(data["finish_rank"], errors="coerce") <= 3).astype(float)
        base_win = float(data["_is_win"].mean()) if len(data) > 0 else 0.075
        base_place = float(data["_is_place"].mean()) if len(data) > 0 else 0.225

        if "horse_name" in data.columns:
            tmp_h = data[["date", "race_id", "horse_name", "jockey_clean"]].copy()
            tmp_h["_h_norm"] = tmp_h["horse_name"].fillna("").astype(str).str.replace(r"\s+", "", regex=True)
            tmp_h = tmp_h.sort_values(["_h_norm", "date", "race_id"])
            shifted_j = tmp_h.groupby("_h_norm")["jockey_clean"].shift(1).fillna("UNKNOWN")
            shifted_j = shifted_j.where(tmp_h["_h_norm"] != "", "UNKNOWN")
            data.loc[tmp_h.index, "prev1_jockey_clean"] = shifted_j

        data["jockey_win_rate"] = _compute_expanding_bayes_rate(data, ["jockey_clean"], "_is_win", base_win, 35.0)
        data["jockey_place_rate"] = _compute_expanding_bayes_rate(data, ["jockey_clean"], "_is_place", base_place, 35.0)
        data["jockey_surface_win_rate"] = _compute_expanding_bayes_rate(
            data, ["jockey_clean", "surface"], "_is_win", base_win, 25.0
        )
        data["jockey_venue_win_rate"] = _compute_expanding_bayes_rate(
            data, ["jockey_clean", "venue"], "_is_win", base_win, 20.0
        )
        data["trainer_win_rate"] = _compute_expanding_bayes_rate(data, ["trainer_clean"], "_is_win", base_win, 35.0)
        data["trainer_place_rate"] = _compute_expanding_bayes_rate(data, ["trainer_clean"], "_is_place", base_place, 35.0)

        j_day_rates = (
            data.groupby(["date", "jockey_clean"], as_index=False)["jockey_win_rate"]
            .first()
            .rename(columns={"jockey_clean": "prev1_jockey_clean", "jockey_win_rate": "_prev_j_rate"})
        )
        data = data.merge(j_day_rates, on=["date", "prev1_jockey_clean"], how="left")
        prev_rate_arr = np.where(
            data["prev1_jockey_clean"] == "UNKNOWN",
            data["jockey_win_rate"],
            data["_prev_j_rate"].fillna(base_win),
        )
        data["jockey_upgrade"] = data["jockey_win_rate"] - prev_rate_arr

        data = data.drop(columns=["_is_win", "_is_place", "_prev_j_rate"], errors="ignore")
        stats_dict = build_entity_stats_artifact(data)
    else:
        stats_dict = (category_maps or {}).get("stats") or {}
        base_win = float(stats_dict.get("base_win", 0.075))
        base_place = float(stats_dict.get("base_place", 0.225))
        jw_map = stats_dict.get("jockey_win", {})
        jp_map = stats_dict.get("jockey_place", {})
        js_map = stats_dict.get("jockey_surface_win", {})
        jv_map = stats_dict.get("jockey_venue_win", {})
        tw_map = stats_dict.get("trainer_win", {})
        tp_map = stats_dict.get("trainer_place", {})

        data["jockey_win_rate"] = data["jockey_clean"].map(jw_map).fillna(base_win).astype(float)
        data["jockey_place_rate"] = data["jockey_clean"].map(jp_map).fillna(base_place).astype(float)

        js_keys = data["jockey_clean"] + "|" + data["surface"]
        data["jockey_surface_win_rate"] = js_keys.map(js_map).fillna(data["jockey_win_rate"]).astype(float)

        jv_keys = data["jockey_clean"] + "|" + data["venue"]
        data["jockey_venue_win_rate"] = jv_keys.map(jv_map).fillna(data["jockey_win_rate"]).astype(float)

        data["trainer_win_rate"] = data["trainer_clean"].map(tw_map).fillna(base_win).astype(float)
        data["trainer_place_rate"] = data["trainer_clean"].map(tp_map).fillna(base_place).astype(float)

        prev_rates = data["prev1_jockey_clean"].map(jw_map).fillna(base_win).astype(float)
        prev_rates = np.where(data["prev1_jockey_clean"] == "UNKNOWN", data["jockey_win_rate"], prev_rates)
        data["jockey_upgrade"] = data["jockey_win_rate"] - prev_rates

    race_avg_jwin = data.groupby("race_id")["jockey_win_rate"].transform("mean")
    data["rel_jockey_win_rate"] = data["jockey_win_rate"] - race_avg_jwin

    if category_maps is None:
        new_maps = {"stats": stats_dict}
        for col in CATEGORICAL_COLS:
            cats = sorted(str(x) for x in set(data[col].unique().tolist()) | {"UNKNOWN"})
            new_maps[col] = cats
            data[col] = pd.Categorical(data[col], categories=cats)
        new_maps["jockey_id"] = sorted(str(x) for x in set(data["jockey_clean"].unique().tolist()))
        new_maps["trainer_id"] = sorted(str(x) for x in set(data["trainer_clean"].unique().tolist()))
        category_maps = new_maps
    else:
        for col in CATEGORICAL_COLS:
            cats = [str(x) for x in category_maps.get(col, ["UNKNOWN"])]
            if "UNKNOWN" not in cats:
                cats.append("UNKNOWN")
            mapped = data[col].where(data[col].isin(cats), "UNKNOWN")
            data[col] = pd.Categorical(mapped, categories=cats)

    return data, category_maps


def calibrate_race_probabilities(
    raw_probs: np.ndarray,
    race_ids: np.ndarray | pd.Series,
    temperature: float = 0.68,
    tail_alpha: float = 1.22,
) -> np.ndarray:
    """
    LightGBMの生出力確率に対して、レース単位でロジット温度スケーリングと
    低確率テール減衰（大穴の勝率インフレ抑制）を適用し、レース内合計1.0に正規化する。
    """
    p_clip = np.clip(np.asarray(raw_probs, dtype=float), 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clip / (1.0 - p_clip))

    adj_logits = np.where(logits < -2.0, -2.0 + (logits + 2.0) * float(tail_alpha), logits)
    scaled = adj_logits / max(float(temperature), 0.10)

    rids = np.asarray(race_ids).astype(str)
    if len(np.unique(rids)) == 1:
        shifted = scaled - np.max(scaled)
        exp_s = np.exp(shifted)
        return exp_s / np.maximum(exp_s.sum(), 1e-12)

    df_tmp = pd.DataFrame({"rid": rids, "s": scaled})
    max_s = df_tmp.groupby("rid")["s"].transform("max").to_numpy()
    exp_s = np.exp(scaled - max_s)
    df_tmp["e"] = exp_s
    sum_e = df_tmp.groupby("rid")["e"].transform("sum").to_numpy()
    return exp_s / np.maximum(sum_e, 1e-12)


def fit_race_calibration(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    race_ids: np.ndarray | pd.Series,
) -> dict:
    """
    時系列検証データの前半を用いてレース内LogLossを最小化する
    (temperature, tail_alpha) を学習する。
    """
    rids = np.asarray(race_ids).astype(str)
    unique_rids = pd.unique(rids)
    split_n = max(1, len(unique_rids) // 2)
    fit_set = set(unique_rids[:split_n])
    mask = np.isin(rids, list(fit_set))

    rp_fit = np.asarray(raw_probs, dtype=float)[mask]
    yt_fit = np.asarray(y_true, dtype=float)[mask]
    rid_fit = rids[mask]

    best_loss = 1e9
    best_t = 0.68
    best_a = 1.22

    for t in np.linspace(0.35, 1.15, 17):
        for a in (1.00, 1.15, 1.30, 1.50):
            cal = calibrate_race_probabilities(rp_fit, rid_fit, temperature=float(t), tail_alpha=float(a))
            cal_c = np.clip(cal, 1e-7, 1.0 - 1e-7)
            loss = float(-np.mean(yt_fit * np.log(cal_c) + (1.0 - yt_fit) * np.log(1.0 - cal_c)))
            if loss < best_loss:
                best_loss = loss
                best_t = float(t)
                best_a = float(a)

    for t in np.linspace(max(0.25, best_t - 0.08), min(1.30, best_t + 0.08), 17):
        for a in np.linspace(max(1.00, best_a - 0.12), min(1.80, best_a + 0.12), 7):
            cal = calibrate_race_probabilities(rp_fit, rid_fit, temperature=float(t), tail_alpha=float(a))
            cal_c = np.clip(cal, 1e-7, 1.0 - 1e-7)
            loss = float(-np.mean(yt_fit * np.log(cal_c) + (1.0 - yt_fit) * np.log(1.0 - cal_c)))
            if loss < best_loss:
                best_loss = loss
                best_t = float(t)
                best_a = float(a)

    return {"temperature": round(float(best_t), 4), "tail_alpha": round(float(best_a), 4)}


def predict_calibrated_win_probs(race_df: pd.DataFrame, artifact: dict) -> np.ndarray:
    """
    推論用共通関数：出馬表DataFrameとモデルartifactからキャリブレーション済みの
    純粋AI単勝予測確率（合計1.0）を算出して返す。
    """
    model = artifact["model"]
    f_cols = artifact["feature_cols"]
    cat_maps = artifact["category_maps"]
    cal_cfg = artifact.get("calibration") or {"temperature": 0.68, "tail_alpha": 1.22}

    feat_df, _ = create_features(race_df, category_maps=cat_maps, is_training=False)
    raw_probs = model.predict_proba(feat_df[f_cols])[:, 1]
    rids = race_df["race_id"].astype(str) if "race_id" in race_df.columns else np.repeat("SINGLE_RACE", len(race_df))
    return calibrate_race_probabilities(
        raw_probs=raw_probs,
        race_ids=rids,
        temperature=float(cal_cfg.get("temperature", 0.68)),
        tail_alpha=float(cal_cfg.get("tail_alpha", 1.22)),
    )


def main():
    if lgb is None:
        raise ImportError("学習を実行するには lightgbm のインストールが必要です (pip install lightgbm)。")

    hist_path = os.path.join("data", "historical_races.csv")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"{hist_path} が見つかりません。先に 05_scrape_history.py でデータを収集してください。")

    raw_df = pd.read_csv(hist_path, dtype={"race_id": str})
    df = normalize_raw_columns(raw_df)
    if "finish_rank" not in df.columns:
        raise KeyError(f"着順列が見つかりません。CSVの実際の列名: {list(raw_df.columns)}")

    df["finish_rank"] = pd.to_numeric(df["finish_rank"], errors="coerce")
    df = df.dropna(subset=["finish_rank"]).copy()
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce").fillna(25.0)
    df = df.sort_values(["date", "race_id", "horse_num"]).reset_index(drop=True)

    n_total_races = int(df["race_id"].nunique())
    print(f"読み込み完了: {n_total_races:,} レース / {len(df):,} 行")

    df["target"] = (df["finish_rank"] == 1).astype(int)

    full_feat, _ = create_features(df, category_maps=None, is_training=True)

    unique_races = full_feat["race_id"].drop_duplicates().tolist()
    split_idx = int(len(unique_races) * 0.85)
    train_races = set(unique_races[:split_idx])
    val_races = set(unique_races[split_idx:])

    train_mask = full_feat["race_id"].isin(train_races)
    val_mask = full_feat["race_id"].isin(val_races)

    X_train = full_feat[train_mask].copy()
    X_val = full_feat[val_mask].copy()
    y_train = X_train["target"].to_numpy()
    y_val = X_val["target"].to_numpy()

    cat_maps = {"stats": build_entity_stats_artifact(df)}
    for col in CATEGORICAL_COLS:
        cats = sorted(str(x) for x in set(full_feat[col].astype(str).unique().tolist()) | {"UNKNOWN"})
        cat_maps[col] = cats
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats)
        X_val[col] = pd.Categorical(X_val[col].astype(str), categories=cats)
    cat_maps["jockey_id"] = sorted(str(x) for x in set(full_feat["jockey_clean"].unique().tolist()))
    cat_maps["trainer_id"] = sorted(str(x) for x in set(full_feat["trainer_clean"].unique().tolist()))

    print("AIモデルを学習中（騎手実績・コース適性・乗り替わり強化特徴量）...")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=45,
        subsample_for_bin=50000,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.2,
        reg_lambda=2.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train[FEATURE_COLS],
        y_train,
        eval_set=[(X_val[FEATURE_COLS], y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
    )

    raw_val_probs = model.predict_proba(X_val[FEATURE_COLS])[:, 1]
    auc = roc_auc_score(y_val, raw_val_probs)

    cal_params = fit_race_calibration(raw_val_probs, y_val, X_val["race_id"].to_numpy())
    X_val["raw_p"] = raw_val_probs
    X_val["naive_p"] = X_val["raw_p"] / X_val.groupby("race_id")["raw_p"].transform("sum")
    X_val["cal_p"] = calibrate_race_probabilities(
        raw_val_probs,
        X_val["race_id"].to_numpy(),
        temperature=cal_params["temperature"],
        tail_alpha=cal_params["tail_alpha"],
    )

    val_unique = pd.unique(X_val["race_id"])
    eval_rids = set(val_unique[len(val_unique) // 2 :])
    eval_df = X_val[X_val["race_id"].isin(eval_rids)].copy()

    ll_naive = log_loss(eval_df["target"], np.clip(eval_df["naive_p"], 1e-7, 1.0 - 1e-7))
    ll_cal = log_loss(eval_df["target"], np.clip(eval_df["cal_p"], 1e-7, 1.0 - 1e-7))

    print(f"検証結果（直近 {len(val_races):,} レース）: ROC-AUC = {auc:.4f}")
    print(f"キャリブレーション: temperature={cal_params['temperature']}, tail_alpha={cal_params['tail_alpha']}")
    print(f"未知レース後半（{len(eval_rids):,}R）LogLoss: 補正前 {ll_naive:.4f} -> 補正後 {ll_cal:.4f}")

    booster_str = str(model.booster_.model_to_string())
    os.makedirs("models", exist_ok=True)
    artifact = {
        "booster_str": booster_str,
        "feature_cols": [str(c) for c in FEATURE_COLS],
        "category_maps": cat_maps,
        "calibration": {
            "temperature": float(cal_params["temperature"]),
            "tail_alpha": float(cal_params["tail_alpha"]),
        },
        "n_races": int(n_total_races),
    }
    out_path = os.path.join("models", "keiba_lgbm.pkl")
    joblib.dump(artifact, out_path, protocol=4)
    print(f"クロスプラットフォーム対応モデルを {out_path} に保存しました。")


if __name__ == "__main__":
    main()