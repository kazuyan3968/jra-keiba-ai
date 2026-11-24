import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

CATEGORICAL_COLS = ["venue", "surface", "track_condition", "jockey_id", "trainer_id"]

FEATURE_COLS = [
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
    "prev1_rank",
    "prev2_rank",
    "prev3_rank",
    "prev1_time_diff",
    "avg_prev_rank",
    "best_prev_rank",
    "rank_trend",
    "rel_prev_rank",
    "rel_time_diff",
]


def create_features(df: pd.DataFrame, category_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """学習・推論共通の特徴量エンジニアリングパイプライン（オッズは特徴量に含めない）"""
    data = df.copy()

    # 1. 馬体重・斤量関連
    data["abs_weight_diff"] = data["weight_diff"].abs()
    data["weight_ratio"] = data["weight_carried"] / data["horse_weight"]

    # 2. 過去走の集計・トレンド
    data["avg_prev_rank"] = data[["prev1_rank", "prev2_rank", "prev3_rank"]].mean(axis=1)
    data["best_prev_rank"] = data[["prev1_rank", "prev2_rank", "prev3_rank"]].min(axis=1)
    data["rank_trend"] = data["prev3_rank"] - data["prev1_rank"]

    # 3. 出走メンバー内での相対評価（レース単位の平均との差）
    race_mean_rank = data.groupby("race_id")["avg_prev_rank"].transform("mean")
    race_mean_tdiff = data.groupby("race_id")["prev1_time_diff"].transform("mean")
    data["rel_prev_rank"] = data["avg_prev_rank"] - race_mean_rank
    data["rel_time_diff"] = data["prev1_time_diff"] - race_mean_tdiff

    # 4. カテゴリ変数の型統一（最大250カテゴリまで保持）
    if category_maps is None:
        category_maps = {}
        for col in CATEGORICAL_COLS:
            top_cats = data[col].value_counts().head(250).index.tolist()
            data[col] = pd.Categorical(data[col], categories=top_cats)
            category_maps[col] = top_cats
    else:
        for col in CATEGORICAL_COLS:
            data[col] = pd.Categorical(data[col], categories=category_maps[col])

    return data, category_maps


def train_and_evaluate():
    df = pd.read_csv("data/historical_races.csv")
    df["target"] = (df["finish_rank"] == 1).astype(int)

    # レースを時系列順に並べ、前半80%を学習用・後半20%を検証用に自動分割
    sorted_races = (
        df[["race_id", "date"]]
        .drop_duplicates()
        .sort_values(["date", "race_id"])
        .reset_index(drop=True)
    )
    total_races = len(sorted_races)
    if total_races < 10:
        raise ValueError(f"レース数が不足しています（現在 {total_races} レース）。10レース以上収集してください。")

    split_idx = max(1, int(total_races * 0.8))
    train_race_ids = set(sorted_races.iloc[:split_idx]["race_id"])
    valid_race_ids = set(sorted_races.iloc[split_idx:]["race_id"])

    train_raw = df[df["race_id"].isin(train_race_ids)].copy()
    valid_raw = df[df["race_id"].isin(valid_race_ids)].copy()

    train_df, category_maps = create_features(train_raw)
    valid_df, _ = create_features(valid_raw, category_maps=category_maps)

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_valid, y_valid = valid_df[FEATURE_COLS], valid_df["target"]

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=400,
        learning_rate=0.03,
        max_leaf_nodes=23,
        categorical_features="from_dtype",
        early_stopping=True,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # レース内で予測勝率の合計が1.0になるよう正規化
    valid_df["raw_prob"] = model.predict_proba(X_valid)[:, 1]
    valid_df["pred_win_prob"] = valid_df["raw_prob"] / valid_df.groupby("race_id")["raw_prob"].transform("sum")
    valid_df["expected_value"] = valid_df["pred_win_prob"] * valid_df["win_odds"]

    auc = roc_auc_score(y_valid, valid_df["pred_win_prob"])
    loss = log_loss(y_valid, valid_df["pred_win_prob"])
    print(f"=== 学習完了 ===")
    print(f"学習レース数: {len(train_race_ids)} R ({train_raw['date'].min()} 〜 {train_raw['date'].max()})")
    print(f"検証レース数: {len(valid_race_ids)} R ({valid_raw['date'].min()} 〜 {valid_raw['date'].max()})")
    print(f"ROC-AUC: {auc:.4f} / LogLoss: {loss:.4f}\n")

    print("=== 期待値ベース単勝回収率シミュレーション ===")
    for ev_th in [1.00, 1.10, 1.20, 1.30]:
        bets = valid_df[valid_df["expected_value"] >= ev_th]
        if len(bets) == 0:
            continue
        hits = bets[bets["finish_rank"] == 1]
        total_bet = len(bets) * 100
        total_return = (hits["win_odds"] * 100).sum()
        roi = (total_return / total_bet) * 100
        hit_rate = (len(hits) / len(bets)) * 100
        print(
            f"期待値 >= {ev_th:.2f} | 購入数: {len(bets):4d}点 | "
            f"的中率: {hit_rate:5.1f}% | 単勝回収率: {roi:6.1f}%"
        )

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": FEATURE_COLS, "category_maps": category_maps},
        "models/keiba_lgbm.pkl",
    )
    print("\n学習済みモデルを models/keiba_lgbm.pkl に保存しました。")


if __name__ == "__main__":
    train_and_evaluate()