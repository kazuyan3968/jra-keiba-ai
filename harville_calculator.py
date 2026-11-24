from collections import Counter
from itertools import combinations, permutations, product
import numpy as np
import pandas as pd


def compute_harville_tensor(win_probs: np.ndarray) -> np.ndarray:
    """単勝確率ベクトルから 1着・2着・3着の全順列確率テンソル P(i, j, k) を算出する"""
    p = np.asarray(win_probs, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    p = p / p.sum()

    n = len(p)
    p1 = p[:, None, None]
    p2 = p[None, :, None]
    p3 = p[None, None, :]

    denom1 = np.maximum(1.0 - p1, 1e-9)
    denom2 = np.maximum(1.0 - p1 - p2, 1e-9)

    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing="ij")
    valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)

    tensor = np.where(valid, p1 * (p2 / denom1) * (p3 / denom2), 0.0)
    return tensor


def calculate_horse_place_probs(df: pd.DataFrame, prob_col: str = "win_prob") -> pd.DataFrame:
    """各馬の連対率（2着以内）および3着内率（複勝圏内）をデータフレームに付与する"""
    out = df.copy()
    tensor = compute_harville_tensor(out[prob_col].to_numpy())

    p_1st = tensor.sum(axis=(1, 2))
    p_2nd = tensor.sum(axis=(0, 2))
    p_3rd = tensor.sum(axis=(0, 1))

    out["pred_win_pct"] = p_1st * 100.0
    out["pred_quinella_pct"] = (p_1st + p_2nd) * 100.0
    out["pred_place_pct"] = (p_1st + p_2nd + p_3rd) * 100.0
    return out


def calculate_pair_probabilities(
    df: pd.DataFrame,
    prob_col: str = "win_prob",
    pair_odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """全馬番ペアの馬連・ワイド的中確率と期待値（EV）を算出する"""
    horses = df.reset_index(drop=True)
    tensor = compute_harville_tensor(horses[prob_col].to_numpy())

    mkt_raw = 0.80 / np.clip(horses["win_odds"].to_numpy(), 1.1, 500.0)
    mkt_tensor = compute_harville_tensor(mkt_raw / mkt_raw.sum())
    rng = np.random.default_rng(42)

    records = []
    n = len(horses)
    for i, j in combinations(range(n), 2):
        p_umaren = tensor[i, j, :].sum() + tensor[j, i, :].sum()
        p_wide = (
            tensor[i, j, :].sum()
            + tensor[j, i, :].sum()
            + tensor[i, :, j].sum()
            + tensor[j, :, i].sum()
            + tensor[:, i, j].sum()
            + tensor[:, j, i].sum()
        )

        mkt_p_umaren = mkt_tensor[i, j, :].sum() + mkt_tensor[j, i, :].sum()
        mkt_p_wide = (
            mkt_tensor[i, j, :].sum()
            + mkt_tensor[j, i, :].sum()
            + mkt_tensor[i, :, j].sum()
            + mkt_tensor[j, :, i].sum()
            + mkt_tensor[:, i, j].sum()
            + mkt_tensor[:, j, i].sum()
        )
        noise_u = rng.uniform(0.90, 1.12)
        noise_w = rng.uniform(0.92, 1.10)
        est_umaren_odds = round(float(np.clip((0.775 / max(mkt_p_umaren, 1e-5)) * noise_u, 1.5, 999.9)), 1)
        est_wide_odds = round(float(np.clip((0.775 / max(mkt_p_wide, 1e-5)) * noise_w, 1.1, 300.0)), 1)

        h1_num = int(horses.loc[i, "horse_num"])
        h2_num = int(horses.loc[j, "horse_num"])
        if h1_num > h2_num:
            h1_num, h2_num = h2_num, h1_num
            h1_name, h2_name = horses.loc[j, "horse_name"], horses.loc[i, "horse_name"]
        else:
            h1_name, h2_name = horses.loc[i, "horse_name"], horses.loc[j, "horse_name"]

        records.append({
            "h1": h1_num,
            "h2": h2_num,
            "組み合わせ": f"{h1_num:02d} - {h2_num:02d}",
            "馬名ペア": f"{h1_name} × {h2_name}",
            "馬連的中率(%)": p_umaren * 100.0,
            "ワイド的中率(%)": p_wide * 100.0,
            "馬連オッズ": est_umaren_odds,
            "ワイドオッズ": est_wide_odds,
        })

    pairs_df = pd.DataFrame(records)

    if pair_odds_df is not None and not pair_odds_df.empty:
        odds_clean = pair_odds_df.copy()
        odds_clean["h1"] = odds_clean[["horse_num_1", "horse_num_2"]].min(axis=1).astype(int)
        odds_clean["h2"] = odds_clean[["horse_num_1", "horse_num_2"]].max(axis=1).astype(int)
        merge_cols = ["h1", "h2"]
        if "umaren_odds" in odds_clean.columns:
            merge_cols.append("umaren_odds")
        if "wide_odds" in odds_clean.columns:
            merge_cols.append("wide_odds")

        pairs_df = pairs_df.merge(odds_clean[merge_cols], on=["h1", "h2"], how="left")
        if "umaren_odds" in pairs_df.columns:
            pairs_df["馬連オッズ"] = pairs_df["umaren_odds"].fillna(pairs_df["馬連オッズ"])
            pairs_df.drop(columns=["umaren_odds"], inplace=True)
        if "wide_odds" in pairs_df.columns:
            pairs_df["ワイドオッズ"] = pairs_df["wide_odds"].fillna(pairs_df["ワイドオッズ"])
            pairs_df.drop(columns=["wide_odds"], inplace=True)

    pairs_df["馬連期待値(EV)"] = (pairs_df["馬連的中率(%)"] / 100.0) * pairs_df["馬連オッズ"]
    pairs_df["ワイド期待値(EV)"] = (pairs_df["ワイド的中率(%)"] / 100.0) * pairs_df["ワイドオッズ"]

    return pairs_df


def calculate_exacta_probabilities(
    df: pd.DataFrame,
    prob_col: str = "win_prob",
    exacta_odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """全順列（nP2）の馬単的中確率と期待値（EV）を算出する"""
    horses = df.reset_index(drop=True)
    tensor = compute_harville_tensor(horses[prob_col].to_numpy())

    mkt_raw = 0.80 / np.clip(horses["win_odds"].to_numpy(), 1.1, 500.0)
    mkt_tensor = compute_harville_tensor(mkt_raw / mkt_raw.sum())
    rng = np.random.default_rng(77)

    records = []
    n = len(horses)
    for i, j in permutations(range(n), 2):
        p_exacta = tensor[i, j, :].sum()
        mkt_p_exacta = mkt_tensor[i, j, :].sum()
        noise_e = rng.uniform(0.88, 1.14)
        est_exacta_odds = round(float(np.clip((0.750 / max(mkt_p_exacta, 1e-5)) * noise_e, 2.0, 1999.9)), 1)

        h1_num = int(horses.loc[i, "horse_num"])
        h2_num = int(horses.loc[j, "horse_num"])
        h1_name = str(horses.loc[i, "horse_name"])
        h2_name = str(horses.loc[j, "horse_name"])

        records.append({
            "h1": h1_num,
            "h2": h2_num,
            "組み合わせ": f"{h1_num:02d} → {h2_num:02d}",
            "馬名順列": f"{h1_name} → {h2_name}",
            "馬単的中率(%)": p_exacta * 100.0,
            "馬単オッズ": est_exacta_odds,
        })

    exacta_df = pd.DataFrame(records)

    if exacta_odds_df is not None and not exacta_odds_df.empty and "exacta_odds" in exacta_odds_df.columns:
        odds_clean = exacta_odds_df.copy()
        odds_clean["h1"] = odds_clean["horse_num_1"].astype(int)
        odds_clean["h2"] = odds_clean["horse_num_2"].astype(int)
        exacta_df = exacta_df.merge(odds_clean[["h1", "h2", "exacta_odds"]], on=["h1", "h2"], how="left")
        exacta_df["馬単オッズ"] = exacta_df["exacta_odds"].fillna(exacta_df["馬単オッズ"])
        exacta_df.drop(columns=["exacta_odds"], inplace=True)

    exacta_df["馬単期待値(EV)"] = (exacta_df["馬単的中率(%)"] / 100.0) * exacta_df["馬単オッズ"]
    return exacta_df


def calculate_trio_probabilities(
    df: pd.DataFrame,
    prob_col: str = "win_prob",
    trio_odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """全3頭組み合わせ（nC3）の三連複的中確率と期待値（EV）を算出する"""
    horses = df.reset_index(drop=True)
    tensor = compute_harville_tensor(horses[prob_col].to_numpy())

    mkt_raw = 0.80 / np.clip(horses["win_odds"].to_numpy(), 1.1, 500.0)
    mkt_tensor = compute_harville_tensor(mkt_raw / mkt_raw.sum())
    rng = np.random.default_rng(108)

    records = []
    n = len(horses)
    for i, j, k in combinations(range(n), 3):
        p_trio = (
            tensor[i, j, k]
            + tensor[i, k, j]
            + tensor[j, i, k]
            + tensor[j, k, i]
            + tensor[k, i, j]
            + tensor[k, j, i]
        )
        mkt_p_trio = (
            mkt_tensor[i, j, k]
            + mkt_tensor[i, k, j]
            + mkt_tensor[j, i, k]
            + mkt_tensor[j, k, i]
            + mkt_tensor[k, i, j]
            + mkt_tensor[k, j, i]
        )

        noise_t = rng.uniform(0.88, 1.18)
        est_trio_odds = round(float(np.clip((0.750 / max(mkt_p_trio, 1e-6)) * noise_t, 3.0, 4999.9)), 1)

        items = sorted(
            [
                (int(horses.loc[i, "horse_num"]), str(horses.loc[i, "horse_name"])),
                (int(horses.loc[j, "horse_num"]), str(horses.loc[j, "horse_name"])),
                (int(horses.loc[k, "horse_num"]), str(horses.loc[k, "horse_name"])),
            ],
            key=lambda x: x[0],
        )
        (h1, n1), (h2, n2), (h3, n3) = items

        records.append({
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "組み合わせ": f"{h1:02d} - {h2:02d} - {h3:02d}",
            "馬名トリオ": f"{n1} × {n2} × {n3}",
            "三連複的中率(%)": p_trio * 100.0,
            "三連複オッズ": est_trio_odds,
        })

    trio_df = pd.DataFrame(records)

    if trio_odds_df is not None and not trio_odds_df.empty:
        odds_clean = trio_odds_df.copy()
        sorted_nums = np.sort(
            odds_clean[["horse_num_1", "horse_num_2", "horse_num_3"]].to_numpy(dtype=int),
            axis=1,
        )
        odds_clean["h1"] = sorted_nums[:, 0]
        odds_clean["h2"] = sorted_nums[:, 1]
        odds_clean["h3"] = sorted_nums[:, 2]
        if "trio_odds" in odds_clean.columns:
            trio_df = trio_df.merge(odds_clean[["h1", "h2", "h3", "trio_odds"]], on=["h1", "h2", "h3"], how="left")
            trio_df["三連複オッズ"] = trio_df["trio_odds"].fillna(trio_df["三連複オッズ"])
            trio_df.drop(columns=["trio_odds"], inplace=True)

    trio_df["三連複期待値(EV)"] = (trio_df["三連複的中率(%)"] / 100.0) * trio_df["三連複オッズ"]
    return trio_df


def build_trio_strategies(
    trio_df: pd.DataFrame,
    horses_df: pd.DataFrame,
    ev_threshold: float = 1.15,
    min_prob_pct: float = 0.5,
    max_single_tickets: int = 15,
    max_followers: int = 7,
    exclude_fav1: bool = False,
) -> dict:
    """期待値上位の組み合わせから『単品』『2頭軸流し』『フォーメーション』を自動生成する"""
    fav1_num = int(horses_df.sort_values("win_odds", ascending=True).iloc[0]["horse_num"])

    cand = trio_df[
        (trio_df["三連複期待値(EV)"] >= ev_threshold) & (trio_df["三連複的中率(%)"] >= min_prob_pct)
    ].sort_values("三連複期待値(EV)", ascending=False)

    base_pool = cand if len(cand) >= 5 else trio_df[trio_df["三連複的中率(%)"] >= min_prob_pct].sort_values(
        "三連複期待値(EV)", ascending=False
    ).head(30)
    if base_pool.empty:
        base_pool = trio_df.sort_values("三連複期待値(EV)", ascending=False).head(30)

    single_tickets = cand.head(max_single_tickets).copy()

    horse_ev_score: Counter = Counter()
    pair_ev_score: Counter = Counter()
    for _, row in base_pool.head(40).iterrows():
        h_tuple = (int(row["h1"]), int(row["h2"]), int(row["h3"]))
        ev_weight = float(row["三連複期待値(EV)"]) * (float(row["三連複的中率(%)"]) ** 0.7)
        for h in h_tuple:
            horse_ev_score[h] += ev_weight
        for p_pair in combinations(h_tuple, 2):
            pair_ev_score[tuple(sorted(p_pair))] += ev_weight

    eligible_pairs = [
        (pair, score)
        for pair, score in pair_ev_score.most_common()
        if not (exclude_fav1 and fav1_num in pair)
    ]
    eligible_horses = [
        (h, score)
        for h, score in horse_ev_score.most_common()
        if not (exclude_fav1 and h == fav1_num)
    ]

    two_axis_result = {"axis": [], "followers": [], "tickets_df": pd.DataFrame()}
    if eligible_pairs:
        axis1, axis2 = eligible_pairs[0][0]
        contain_axis = trio_df[
            trio_df.apply(lambda r: {axis1, axis2}.issubset({int(r["h1"]), int(r["h2"]), int(r["h3"])}), axis=1)
        ].copy()
        contain_axis["相手馬番"] = contain_axis.apply(
            lambda r: list({int(r["h1"]), int(r["h2"]), int(r["h3"])} - {axis1, axis2})[0], axis=1
        )
        good_followers = contain_axis[contain_axis["三連複期待値(EV)"] >= ev_threshold].sort_values(
            "三連複期待値(EV)", ascending=False
        )
        if len(good_followers) == 0:
            good_followers = contain_axis.sort_values("三連複期待値(EV)", ascending=False).head(min(4, max_followers))
        else:
            good_followers = good_followers.head(max_followers)

        followers_sorted = sorted(good_followers["相手馬番"].astype(int).tolist())
        axis_tickets = contain_axis[contain_axis["相手馬番"].isin(followers_sorted)].sort_values(
            "三連複期待値(EV)", ascending=False
        )
        two_axis_result = {
            "axis": [axis1, axis2],
            "followers": followers_sorted,
            "tickets_df": axis_tickets.drop(columns=["相手馬番"]),
        }

    formation_result = {"col1": [], "col2": [], "col3": [], "tickets_df": pd.DataFrame()}
    if len(eligible_horses) >= 2 and len(horse_ev_score) >= 4:
        col1 = sorted([h for h, _ in eligible_horses[:2]])
        col2_raw = [h for h, _ in eligible_horses[:4]]
        col2 = sorted(list(dict.fromkeys(col1 + col2_raw)))
        col3_limit = max(5, min(max_followers + 1, len(horses_df)))
        col3_raw = [h for h, _ in horse_ev_score.most_common(col3_limit)]
        col3 = sorted(list(dict.fromkeys(col2 + col3_raw)))

        valid_trios = set()
        for a, b, c in product(col1, col2, col3):
            if len({a, b, c}) == 3:
                valid_trios.add(tuple(sorted((a, b, c))))

        if valid_trios:
            trio_keys = set(valid_trios)
            form_tickets = trio_df[
                trio_df.apply(lambda r: (int(r["h1"]), int(r["h2"]), int(r["h3"])) in trio_keys, axis=1)
            ].sort_values("三連複期待値(EV)", ascending=False)
            formation_result = {
                "col1": col1,
                "col2": col2,
                "col3": col3,
                "tickets_df": form_tickets,
            }

    return {
        "fav1_num": fav1_num,
        "single_tickets": single_tickets,
        "two_axis": two_axis_result,
        "formation": formation_result,
    }


def _select_top_horses_safe(
    ranked_horses: list[int],
    fallback_order: list[int],
    count: int,
    exclude_num: int | None = None,
) -> list[int]:
    """指定頭数の馬番リストを安全に選出する"""
    merged = list(dict.fromkeys(ranked_horses + fallback_order))
    if exclude_num is not None:
        filtered = [h for h in merged if h != exclude_num]
        if len(filtered) >= count:
            return sorted(filtered[:count])
    return sorted(merged[:count])


def build_budget_mode_tickets(
    horses_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    exacta_df: pd.DataFrame,
    trio_df: pd.DataFrame,
    exclude_fav1: bool = False,
    style_mode: str = "balanced",
) -> dict:
    """
    style_mode:
      - 'solid'    : 堅実重視（的中率の重みを強くし、上位実力馬を中心に構成）
      - 'balanced' : 標準バランス（1頭目軸は実力上位馬、2頭目軸と相手に妙味中穴を絡める）
      - 'upset'    : 穴狙い特化（期待値EVを最優先して高配当狙い）
    """
    fav1_num = int(horses_df.sort_values("win_odds", ascending=True).iloc[0]["horse_num"])
    sorted_by_prob = horses_df.sort_values("pred_win_pct", ascending=False).reset_index(drop=True)
    fallback_horses = sorted_by_prob["horse_num"].astype(int).tolist()

    # 実力圏内（AI予測勝率上位8頭、または予測勝率4%以上）を定義
    capable_set = set(
        sorted_by_prob[
            (sorted_by_prob.index < 8) | (sorted_by_prob["pred_win_pct"] >= 4.0)
        ]["horse_num"].astype(int).tolist()
    )

    # スタイルに応じた確率べき乗の重み（高いほど的中率重視、低いほど高オッズEV重視）
    if style_mode == "solid":
        prob_power = 1.25
        min_trio_p = 0.60
    elif style_mode == "upset":
        prob_power = 0.45
        min_trio_p = 0.25
    else:
        prob_power = 0.85
        min_trio_p = 0.40

    # --- 1. 三連複 2頭軸5頭流し（500円モード ＆ 2000円モード共通） ---
    trio_pool = trio_df[trio_df["三連複的中率(%)"] >= min_trio_p].copy()
    if len(trio_pool) < 15:
        trio_pool = trio_df.sort_values("三連複的中率(%)", ascending=False).head(60).copy()

    trio_pool["score"] = trio_pool["三連複期待値(EV)"] * (np.maximum(trio_pool["三連複的中率(%)"], 0.01) ** prob_power)
    top_trios = trio_pool.sort_values("score", ascending=False).head(50)

    pair_score: Counter = Counter()
    for _, r in top_trios.iterrows():
        t_nums = (int(r["h1"]), int(r["h2"]), int(r["h3"]))
        w = float(r["score"])
        for p in combinations(t_nums, 2):
            pair_score[tuple(sorted(p))] += w

    # 軸ペアの選定
    eligible_pairs = []
    for pair, score in pair_score.most_common():
        if exclude_fav1 and fav1_num in pair:
            continue
        if style_mode == "solid":
            # 堅実モード: 軸2頭とも実力圏内
            if pair[0] in capable_set and pair[1] in capable_set:
                eligible_pairs.append((pair, score))
        elif style_mode == "balanced":
            # 標準バランス: 少なくとも1頭はAI予測上位4頭以内の本命・対抗格を含める
            top4_set = set(fallback_horses[:4])
            if (pair[0] in top4_set or pair[1] in top4_set) and (pair[0] in capable_set and pair[1] in capable_set):
                eligible_pairs.append((pair, score))
        else:
            eligible_pairs.append((pair, score))

    if not eligible_pairs:
        eligible_pairs = [
            (pair, score) for pair, score in pair_score.most_common()
            if not (exclude_fav1 and fav1_num in pair)
        ]

    if eligible_pairs:
        axis1, axis2 = eligible_pairs[0][0]
    else:
        safe_axes = _select_top_horses_safe([], fallback_horses, 2, fav1_num if exclude_fav1 else None)
        axis1, axis2 = safe_axes[0], safe_axes[1]

    contain_axis = trio_df[
        trio_df.apply(lambda r: {axis1, axis2}.issubset({int(r["h1"]), int(r["h2"]), int(r["h3"])}), axis=1)
    ].copy()
    contain_axis["相手馬番"] = contain_axis.apply(
        lambda r: list({int(r["h1"]), int(r["h2"]), int(r["h3"])} - {axis1, axis2})[0], axis=1
    )
    contain_axis["follower_score"] = contain_axis["三連複期待値(EV)"] * (
        np.maximum(contain_axis["三連複的中率(%)"], 0.01) ** prob_power
    )
    top5_followers_df = contain_axis.sort_values("follower_score", ascending=False).head(5)
    followers_5 = sorted(top5_followers_df["相手馬番"].astype(int).tolist())
    trio_2axis_5_df = contain_axis[contain_axis["相手馬番"].isin(followers_5)].sort_values(
        "三連複期待値(EV)", ascending=False
    ).copy()
    trio_2axis_5_df["券種"] = "三連複(2頭軸流し)"
    trio_2axis_5_df["購入額(円)"] = 100
    trio_2axis_5_df["想定払戻(円)"] = (trio_2axis_5_df["三連複オッズ"] * 100).astype(int)

    # --- 2. 300円モード（極端な大穴を除外し、現実的な妙味トップ3を選出） ---
    cross_candidates = []
    min_win_cut = 5.0 if style_mode == "solid" else (3.5 if style_mode == "balanced" else 2.5)
    min_wide_cut = 3.0 if style_mode == "solid" else (2.0 if style_mode == "balanced" else 1.2)
    min_umaren_cut = 2.0 if style_mode == "solid" else (1.2 if style_mode == "balanced" else 0.8)
    min_exacta_cut = 1.5 if style_mode == "solid" else (0.9 if style_mode == "balanced" else 0.6)
    min_trio_cut = 0.8 if style_mode == "solid" else (0.45 if style_mode == "balanced" else 0.25)

    for _, r in horses_df[horses_df["pred_win_pct"] >= min_win_cut].iterrows():
        h_num = int(r["horse_num"])
        if exclude_fav1 and h_num == fav1_num:
            continue
        ev = (float(r["pred_win_pct"]) / 100.0) * float(r["win_odds"])
        cross_candidates.append({
            "券種": "単勝",
            "組み合わせ": f"{h_num:02d}",
            "対象馬名": str(r["horse_name"]),
            "オッズ": float(r["win_odds"]),
            "的中率(%)": float(r["pred_win_pct"]),
            "期待値(EV)": ev,
            "score": ev * (float(r["pred_win_pct"]) ** (prob_power * 0.4)),
            "購入額(円)": 100,
            "想定払戻(円)": int(float(r["win_odds"]) * 100),
        })

    for _, r in pairs_df[pairs_df["ワイド的中率(%)"] >= min_wide_cut].iterrows():
        if exclude_fav1 and fav1_num in (int(r["h1"]), int(r["h2"])):
            continue
        ev = float(r["ワイド期待値(EV)"])
        p = float(r["ワイド的中率(%)"])
        cross_candidates.append({
            "券種": "ワイド",
            "組み合わせ": r["組み合わせ"],
            "対象馬名": r["馬名ペア"],
            "オッズ": float(r["ワイドオッズ"]),
            "的中率(%)": p,
            "期待値(EV)": ev,
            "score": ev * (p ** (prob_power * 0.4)),
            "購入額(円)": 100,
            "想定払戻(円)": int(float(r["ワイドオッズ"]) * 100),
        })

    for _, r in pairs_df[pairs_df["馬連的中率(%)"] >= min_umaren_cut].iterrows():
        if exclude_fav1 and fav1_num in (int(r["h1"]), int(r["h2"])):
            continue
        ev = float(r["馬連期待値(EV)"])
        p = float(r["馬連的中率(%)"])
        cross_candidates.append({
            "券種": "馬連",
            "組み合わせ": r["組み合わせ"],
            "対象馬名": r["馬名ペア"],
            "オッズ": float(r["馬連オッズ"]),
            "的中率(%)": p,
            "期待値(EV)": ev,
            "score": ev * (p ** (prob_power * 0.4)),
            "購入額(円)": 100,
            "想定払戻(円)": int(float(r["馬連オッズ"]) * 100),
        })

    for _, r in exacta_df[exacta_df["馬単的中率(%)"] >= min_exacta_cut].iterrows():
        if exclude_fav1 and int(r["h1"]) == fav1_num:
            continue
        ev = float(r["馬単期待値(EV)"])
        p = float(r["馬単的中率(%)"])
        cross_candidates.append({
            "券種": "馬単",
            "組み合わせ": r["組み合わせ"],
            "対象馬名": r["馬名順列"],
            "オッズ": float(r["馬単オッズ"]),
            "的中率(%)": p,
            "期待値(EV)": ev,
            "score": ev * (p ** (prob_power * 0.4)),
            "購入額(円)": 100,
            "想定払戻(円)": int(float(r["馬単オッズ"]) * 100),
        })

    for _, r in trio_df[trio_df["三連複的中率(%)"] >= min_trio_cut].iterrows():
        if exclude_fav1 and fav1_num in (int(r["h1"]), int(r["h2"]), int(r["h3"])):
            continue
        ev = float(r["三連複期待値(EV)"])
        p = float(r["三連複的中率(%)"])
        cross_candidates.append({
            "券種": "三連複",
            "組み合わせ": r["組み合わせ"],
            "対象馬名": r["馬名トリオ"],
            "オッズ": float(r["三連複オッズ"]),
            "的中率(%)": p,
            "期待値(EV)": ev,
            "score": ev * (p ** (prob_power * 0.4)),
            "購入額(円)": 100,
            "想定払戻(円)": int(float(r["三連複オッズ"]) * 100),
        })

    if len(cross_candidates) < 3:
        for _, r in sorted_by_prob.head(5).iterrows():
            h_num = int(r["horse_num"])
            ev = (float(r["pred_win_pct"]) / 100.0) * float(r["win_odds"])
            cross_candidates.append({
                "券種": "単勝",
                "組み合わせ": f"{h_num:02d}",
                "対象馬名": str(r["horse_name"]),
                "オッズ": float(r["win_odds"]),
                "的中率(%)": float(r["pred_win_pct"]),
                "期待値(EV)": ev,
                "score": ev,
                "購入額(円)": 100,
                "想定払戻(円)": int(float(r["win_odds"]) * 100),
            })

    mode_300_df = (
        pd.DataFrame(cross_candidates)
        .drop_duplicates(subset=["券種", "組み合わせ"])
        .sort_values("score", ascending=False)
        .head(3)
        .drop(columns=["score"])
        .reset_index(drop=True)
    )

    # --- 3. 2000円モード用：馬単 4頭ボックス（12点 × 100円 = 1200円） ---
    top_exacta = exacta_df[exacta_df["馬単的中率(%)"] >= min_exacta_cut].copy()
    if top_exacta.empty:
        top_exacta = exacta_df.sort_values("馬単的中率(%)", ascending=False).head(30).copy()
    top_exacta["score"] = top_exacta["馬単期待値(EV)"] * (np.maximum(top_exacta["馬単的中率(%)"], 0.01) ** prob_power)

    exacta_horse_score: Counter = Counter()
    for _, r in top_exacta.sort_values("score", ascending=False).head(30).iterrows():
        w_e = float(r["score"])
        exacta_horse_score[int(r["h1"])] += w_e * 1.2
        exacta_horse_score[int(r["h2"])] += w_e

    box4_pool = [
        h for h, _ in exacta_horse_score.most_common()
        if (style_mode == "upset" or h in capable_set)
    ]
    box4_nums = _select_top_horses_safe(
        box4_pool,
        fallback_horses,
        min(4, len(fallback_horses)),
        fav1_num if exclude_fav1 else None,
    )

    exacta_box_perms = set(permutations(box4_nums, 2))
    exacta_box4_df = exacta_df[
        exacta_df.apply(lambda r: (int(r["h1"]), int(r["h2"])) in exacta_box_perms, axis=1)
    ].sort_values("馬単期待値(EV)", ascending=False).copy()
    exacta_box4_df["券種"] = "馬単(4頭BOX)"
    exacta_box4_df["購入額(円)"] = 100
    exacta_box4_df["想定払戻(円)"] = (exacta_box4_df["馬単オッズ"] * 100).astype(int)

    # --- 4. 2000円モード用：ワイド 3頭ボックス（3点 × 100円 = 300円） ---
    top_wide = pairs_df[pairs_df["ワイド的中率(%)"] >= min_wide_cut].copy()
    if top_wide.empty:
        top_wide = pairs_df.sort_values("ワイド的中率(%)", ascending=False).head(25).copy()
    top_wide["score"] = top_wide["ワイド期待値(EV)"] * (np.maximum(top_wide["ワイド的中率(%)"], 0.01) ** prob_power)

    wide_horse_score: Counter = Counter()
    for _, r in top_wide.sort_values("score", ascending=False).head(25).iterrows():
        w_w = float(r["score"])
        wide_horse_score[int(r["h1"])] += w_w
        wide_horse_score[int(r["h2"])] += w_w

    box3_pool = [
        h for h, _ in wide_horse_score.most_common()
        if (style_mode == "upset" or h in capable_set)
    ]
    box3_nums = _select_top_horses_safe(
        box3_pool,
        fallback_horses,
        min(3, len(fallback_horses)),
        fav1_num if exclude_fav1 else None,
    )

    wide_box_pairs = {tuple(sorted(p)) for p in combinations(box3_nums, 2)}
    wide_box3_df = pairs_df[
        pairs_df.apply(lambda r: (int(r["h1"]), int(r["h2"])) in wide_box_pairs, axis=1)
    ].sort_values("ワイド期待値(EV)", ascending=False).copy()
    wide_box3_df["券種"] = "ワイド(3頭BOX)"
    wide_box3_df["購入額(円)"] = 100
    wide_box3_df["想定払戻(円)"] = (wide_box3_df["ワイドオッズ"] * 100).astype(int)

    return {
        "fav1_num": fav1_num,
        "mode_500": {
            "axis": [axis1, axis2],
            "followers": followers_5,
            "tickets_df": trio_2axis_5_df,
            "total_cost": len(trio_2axis_5_df) * 100,
        },
        "mode_300": {
            "tickets_df": mode_300_df,
            "total_cost": len(mode_300_df) * 100,
        },
        "mode_2000": {
            "trio_axis": [axis1, axis2],
            "trio_followers": followers_5,
            "trio_df": trio_2axis_5_df,
            "exacta_box": box4_nums,
            "exacta_df": exacta_box4_df,
            "wide_box": box3_nums,
            "wide_df": wide_box3_df,
            "total_cost": (len(trio_2axis_5_df) + len(exacta_box4_df) + len(wide_box3_df)) * 100,
        },
    }