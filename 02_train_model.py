import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

DATA_PATH = "data/historical_races.csv"
MODEL_PATH = "models/keiba_lgbm.pkl"

CAT_COLS = ["venue", "surface", "track_condition", "jockey_id", "trainer_id"]

FEATURE_COLS = [
    "prev1_rank",
    "prev2_rank",
    "prev3_rank",
    "avg_prev_rank",
    "best_prev_rank",
    "rank_trend",
    "prev1_time_diff",
    "rel_prev_rank",
    "rel_time_diff",
    "venue",
    "surface",
    "distance",
    "track_condition",
    "frame_num",
    "horse_num",
    "weight_carried",
    "horse_weight",
    "weight_diff",
    "abs_weight_diff",
    "weight_ratio",
    "jockey_id",
    "trainer_id",
]


def create_features(df: pd.DataFrame, category_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """学習時・推論時で共通の22特徴量を生成する（オッズ・人気順は除外）"""
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

    # 1. 過去3走の成績・上昇トレンド
    out["avg_prev_rank"] = (out["prev1_rank"] + out["prev2_rank"] + out["prev3_rank"]) / 3.0
    out["best_prev_rank"] = out[["prev1_rank", "prev2_rank", "prev3_rank"]].min(axis=1)
    out["rank_trend"] = out["prev3_rank"] - out["prev1_rank"]

    # 2. 出走メンバー内での相対評価
    if "race_id" in out.columns:
        race_avg_rank = out.groupby("race_id")["avg_prev_rank"].transform("mean")
        race_avg_tdiff = out.groupby("race_id")["prev1_time_diff"].transform("mean")
    else:
        race_avg_rank = out["avg_prev_rank"].mean()
        race_avg_tdiff = out["prev1_time_diff"].mean()

    out["rel_prev_rank"] = out["avg_prev_rank"] - race_avg_rank
    out["rel_time_diff"] = out["prev1_time_diff"] - race_avg_tdiff

    # 3. 馬体・斤量バランス
    safe_hw = np.clip(out["horse_weight"].to_numpy(dtype=float), 350.0, 650.0)
    out["abs_weight_diff"] = out["weight_diff"].abs()
    out["weight_ratio"] = out["weight_carried"] / safe_hw

    # 4. カテゴリ変数（上位250カテゴリ + OTHER = 最大251で上限255以内に収める）
    fitted_maps: dict[str, list[str]] = {} if category_maps is None else dict(category_maps)
    for col in CAT_COLS:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        s = out[col].fillna("UNKNOWN").astype(str).str.strip()
        s = s.replace("", "UNKNOWN")

        if category_maps is None:
            top_cats = s.value_counts().head(250).index.tolist()
            if "OTHER" not in top_cats:
                top_cats.append("OTHER")
            fitted_maps[col] = top_cats
        else:
            top_cats = fitted_maps[col]

        s_mapped = s.where(s.isin(top_cats), "OTHER")
        out[col] = pd.Categorical(s_mapped, categories=top_cats)

    return out, fitted_maps


def train_and_evaluate():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} が見つかりません。先に 05_scrape_history.py を実行してください。")

    df = pd.read_csv(DATA_PATH, dtype={"race_id": str})

    # rank / finish_rank のどちらの列名にも対応
    if "finish_rank" not in df.columns and "rank" in df.columns:
        df["finish_rank"] = df["rank"]
    if "finish_rank" not in df.columns:
        raise KeyError("CSV内に着順列（rank または finish_rank）が見つかりません。")

    df["finish_rank"] = pd.to_numeric(df["finish_rank"], errors="coerce")
    df = df.dropna(subset=["finish_rank"]).copy()
    df["target"] = (df["finish_rank"] == 1).astype(int)

    # 時系列順にソート（リーク防止）
    sort_cols = [c for c in ("date", "race_id", "horse_num") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    n_races = int(df["race_id"].nunique()) if "race_id" in df.columns else len(df) // 12
    print(f"読み込み完了: {n_races:,} レース / {len(df):,} 行")

    # 前半85%を学習、直近15%を検証に分割
    unique_races = df["race_id"].unique()
    split_idx = max(1, int(len(unique_races) * 0.85))
    train_races = set(unique_races[:split_idx])
    val_races = set(unique_races[split_idx:])

    train_raw = df[df["race_id"].isin(train_races)].copy()
    val_raw = df[df["race_id"].isin(val_races)].copy()

    train_df, category_maps = create_features(train_raw, category_maps=None)
    val_df, _ = create_features(val_raw, category_maps=category_maps)

    # データ量に応じたハイパーパラメータ自動調整
    if n_races < 3000:
        model = HistGradientBoostingClassifier(
            categorical_features=CAT_COLS,
            max_iter=400,
            learning_rate=0.03,
            max_leaf_nodes=23,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        model = HistGradientBoostingClassifier(
            categorical_features=CAT_COLS,
            max_iter=800,
            learning_rate=0.025,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.5,
            random_state=42,
        )

    print("AIモデルを学習中...")
    model.fit(train_df[FEATURE_COLS], train_df["target"])

    # 検証データでの精度・回収率シミュレーション
    if len(val_df) > 0:
        val_raw_probs = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
        val_df["raw_prob"] = val_raw_probs
        val_df["win_prob"] = val_df["raw_prob"] / val_df.groupby("race_id")["raw_prob"].transform("sum")

        auc = roc_auc_score(val_df["target"], val_df["win_prob"])
        ll = log_loss(val_df["target"], np.clip(val_df["win_prob"], 1e-6, 1.0 - 1e-6))
        print(f"検証結果（直近 {len(val_races):,} レース）: ROC-AUC = {auc:.4f} / LogLoss = {ll:.4f}")

        if "win_odds" in val_df.columns:
            val_df["win_odds"] = pd.to_numeric(val_df["win_odds"], errors="coerce").fillna(0.0)
            val_df["ev"] = val_df["win_prob"] * val_df["win_odds"]
            print("--- 検証セット 単勝期待値(EV)別 シミュレーション ---")
            for th in (1.00, 1.10, 1.15, 1.20, 1.30):
                sub = val_df[(val_df["ev"] >= th) & (val_df["win_odds"] >= 1.1)]
                if len(sub) > 0:
                    hits = int(sub["target"].sum())
                    ret = float((sub["target"] * sub["win_odds"] * 100).sum())
                    cost = len(sub) * 100
                    roi = (ret / cost) * 100.0
                    print(f"  EV >= {th:.2f}: {len(sub):5d}点 / 的中 {hits:3d}回 ({hits/len(sub)*100:5.1f}%) / 回収率 {roi:6.1f}%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    artifact = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "category_maps": category_maps,
        "n_races": n_races,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"学習済みモデルを {MODEL_PATH} に保存しました。")


if __name__ == "__main__":
    train_and_evaluate()