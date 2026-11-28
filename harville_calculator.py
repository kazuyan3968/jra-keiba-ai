import itertools
import numpy as np
import pandas as pd


def calculate_horse_place_probs(race_df: pd.DataFrame, prob_col: str = "win_prob") -> pd.DataFrame:
    """
    ハービル公式により各馬の単勝率・連対率(2着以内)・複勝率(3着以内)を算出する。
    """
    df = race_df.copy()
    p = np.clip(pd.to_numeric(df[prob_col], errors="coerce").fillna(1e-5).to_numpy(dtype=float), 1e-6, 1.0)
    p = p / p.sum()
    n = len(p)

    quinella = p.copy()
    place = p.copy()

    if n >= 2:
        for j in range(n):
            p2 = 0.0
            for i in range(n):
                if i == j:
                    continue
                d1 = max(1e-6, 1.0 - p[i])
                p2 += p[i] * (p[j] / d1)
            quinella[j] = p[j] + p2

    if n >= 3:
        for k in range(n):
            p3 = 0.0
            for i in range(n):
                if i == k:
                    continue
                d1 = max(1e-6, 1.0 - p[i])
                for j in range(n):
                    if j == i or j == k:
                        continue
                    d2 = max(1e-6, 1.0 - p[i] - p[j])
                    p3 += p[i] * (p[j] / d1) * (p[k] / d2)
            place[k] = quinella[k] + p3
    else:
        place = quinella.copy()

    df["pred_win_pct"] = (p * 100.0).round(2)
    df["pred_quinella_pct"] = (np.clip(quinella, 0.0, 0.999) * 100.0).round(2)
    df["pred_place_pct"] = (np.clip(place, 0.0, 0.999) * 100.0).round(2)
    return df


def _build_market_probs(df: pd.DataFrame) -> dict[int, float]:
    nums = df["horse_num"].astype(int).tolist()
    odds = np.clip(pd.to_numeric(df["win_odds"], errors="coerce").fillna(25.0).to_numpy(dtype=float), 1.05, 999.9)
    raw = 1.0 / odds
    q = raw / raw.sum()
    return dict(zip(nums, q))


def calculate_pair_probabilities(
    race_df: pd.DataFrame,
    prob_col: str = "win_prob",
    pair_odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """全組合せの馬連・ワイド確率および実オッズ（未取得時は合成オッズ）期待値を算出する"""
    df = race_df.sort_values("horse_num").reset_index(drop=True)
    nums = df["horse_num"].astype(int).tolist()
    names = dict(zip(nums, df["horse_name"].astype(str)))
    p_arr = np.clip(pd.to_numeric(df[prob_col], errors="coerce").fillna(1e-5).to_numpy(dtype=float), 1e-6, 1.0)
    p_arr = p_arr / p_arr.sum()
    p_map = dict(zip(nums, p_arr))
    q_map = _build_market_probs(df)

    umaren_real = {}
    wide_real = {}
    if pair_odds_df is not None and not pair_odds_df.empty:
        for _, r in pair_odds_df.iterrows():
            h1, h2 = int(r["horse_num_1"]), int(r["horse_num_2"])
            k = (min(h1, h2), max(h1, h2))
            if "umaren_odds" in r and pd.notna(r["umaren_odds"]):
                umaren_real[k] = float(r["umaren_odds"])
            if "wide_odds" in r and pd.notna(r["wide_odds"]):
                wide_real[k] = float(r["wide_odds"])

    rows = []
    for h1, h2 in itertools.combinations(nums, 2):
        p1, p2 = p_map[h1], p_map[h2]
        p_umaren = p1 * (p2 / max(1e-6, 1.0 - p1)) + p2 * (p1 / max(1e-6, 1.0 - p2))

        p_wide = p_umaren
        for h3 in nums:
            if h3 == h1 or h3 == h2:
                continue
            p3 = p_map[h3]
            d3 = max(1e-6, 1.0 - p3)
            p_wide += p3 * (p1 / d3) * (p2 / max(1e-6, d3 - p1))
            p_wide += p3 * (p2 / d3) * (p1 / max(1e-6, d3 - p2))
            d1 = max(1e-6, 1.0 - p1)
            p_wide += p1 * (p3 / d1) * (p2 / max(1e-6, d1 - p3))
            d2 = max(1e-6, 1.0 - p2)
            p_wide += p2 * (p3 / d2) * (p1 / max(1e-6, d2 - p3))

        k = (h1, h2)
        if k in umaren_real:
            o_u = umaren_real[k]
        else:
            q1, q2 = q_map[h1], q_map[h2]
            qu = q1 * (q2 / max(1e-6, 1.0 - q1)) + q2 * (q1 / max(1e-6, 1.0 - q2))
            o_u = round(float(np.clip(0.775 / max(1e-6, qu), 1.4, 999.9)), 1)

        if k in wide_real:
            o_w = wide_real[k]
        else:
            o_w = round(float(np.clip(o_u * 0.36, 1.1, 300.0)), 1)

        rows.append({
            "horse_num_1": h1,
            "horse_num_2": h2,
            "組み合わせ": f"{h1:02d} - {h2:02d}",
            "馬名ペア": f"{names[h1]} × {names[h2]}",
            "馬連オッズ": o_u,
            "馬連的中率(%)": round(p_umaren * 100.0, 2),
            "馬連期待値(EV)": round(p_umaren * o_u, 2),
            "ワイドオッズ": o_w,
            "ワイド的中率(%)": round(min(p_wide, 0.99) * 100.0, 2),
            "ワイド期待値(EV)": round(min(p_wide, 0.99) * o_w, 2),
        })

    return pd.DataFrame(rows)


def calculate_exacta_probabilities(
    race_df: pd.DataFrame,
    prob_col: str = "win_prob",
    exacta_odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """全組合せの馬単確率・期待値を算出する"""
    df = race_df.sort_values("horse_num").reset_index(drop=True)
    nums = df["horse_num"].astype(int).tolist()
    names = dict(zip(nums, df["horse_name"].astype(str)))
    p_arr = np.clip(pd.to_numeric(df[prob_col], errors="coerce").fillna(1e-5).to_numpy(dtype=float), 1e-6, 1.0)
    p_arr = p_arr / p_arr.sum()
    p_map = dict(zip(nums, p_arr))
    q_map = _build_market_probs(df)

    exacta_real = {}
    if exacta_odds_df is not None and not exacta_odds_df.empty:
        for _, r in exacta_odds_df.iterrows():
            k = (int(r["horse_num_1"]), int(r["horse_num_2"]))
            if "exacta_odds" in r and pd.notna(r["exacta_odds"]):
                exacta_real[k] = float(r["exacta_odds"])

    rows = []
    for h1, h2 in itertools.permutations(nums, 2):
        p1, p2 = p_map[h1], p_map[h2]
        p_ex = p1 * (p2 / max(1e-6, 1.0 - p1))
        k = (h1, h2)
        if k in exacta_real:
            o_ex = exacta_real[k]
        else:
            q1, q2 = q_map[h1], q_map[h2]
            q_ex = q1 * (q2 / max(1e-6, 1.0 - q1))
            o_ex = round(float(np.clip(0.75 / max(1e-6, q_ex), 1.8, 1999.9)), 1)

        rows.append({
            "horse_num_1": h1,
            "horse_num_2": h2,
            "組み合わせ": f"{h1:02d} → {h2:02d}",
            "馬名順列": f"{names[h1]} → {names[h2]}",
            "馬単オッズ": o_ex,
            "馬単的中率(%)": round(p_ex * 100.0, 2),
            "馬単期待値(EV)": round(p_ex * o_ex, 2),
        })

    return pd.DataFrame(rows)


def calculate_trio_probabilities(
    race_df: pd.DataFrame,
    prob_col: str = "win_prob",
    trio_odds_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """上位10頭の三連複組合せ確率・期待値を算出する"""
    df_all = race_df.sort_values("horse_num").reset_index(drop=True)
    nums_all = df_all["horse_num"].astype(int).tolist()
    names = dict(zip(nums_all, df_all["horse_name"].astype(str)))
    p_arr = np.clip(pd.to_numeric(df_all[prob_col], errors="coerce").fillna(1e-5).to_numpy(dtype=float), 1e-6, 1.0)
    p_arr = p_arr / p_arr.sum()
    p_map = dict(zip(nums_all, p_arr))
    q_map = _build_market_probs(df_all)

    df_all["_p"] = p_arr
    top_nums = sorted(df_all.sort_values("_p", ascending=False)["horse_num"].astype(int).head(10).tolist())

    trio_real = {}
    if trio_odds_df is not None and not trio_odds_df.empty:
        for _, r in trio_odds_df.iterrows():
            k = tuple(sorted([int(r["horse_num_1"]), int(r["horse_num_2"]), int(r["horse_num_3"])]))
            if "trio_odds" in r and pd.notna(r["trio_odds"]):
                trio_real[k] = float(r["trio_odds"])

    rows = []
    for h1, h2, h3 in itertools.combinations(top_nums, 3):
        p_trio = 0.0
        for a, b, c in itertools.permutations((h1, h2, h3)):
            pa, pb, pc = p_map[a], p_map[b], p_map[c]
            p_trio += pa * (pb / max(1e-6, 1.0 - pa)) * (pc / max(1e-6, 1.0 - pa - pb))

        k = (h1, h2, h3)
        if k in trio_real:
            o_trio = trio_real[k]
        else:
            q_trio = 0.0
            for a, b, c in itertools.permutations((h1, h2, h3)):
                qa, qb, qc = q_map[a], q_map[b], q_map[c]
                q_trio += qa * (qb / max(1e-6, 1.0 - qa)) * (qc / max(1e-6, 1.0 - qa - qb))
            o_trio = round(float(np.clip(0.75 / max(1e-6, q_trio), 2.5, 4999.9)), 1)

        rows.append({
            "horse_num_1": h1,
            "horse_num_2": h2,
            "horse_num_3": h3,
            "組み合わせ": f"{h1:02d} - {h2:02d} - {h3:02d}",
            "馬名トリオ": f"{names[h1]} × {names[h2]} × {names[h3]}",
            "三連複オッズ": o_trio,
            "三連複的中率(%)": round(p_trio * 100.0, 2),
            "三連複期待値(EV)": round(p_trio * o_trio, 2),
        })

    return pd.DataFrame(rows)


def _rank_horses_for_bets(
    horses_df: pd.DataFrame,
    exclude_fav1: bool,
    style_mode: str,
) -> tuple[pd.DataFrame, set[int], set[int]]:
    """
    出走表の「印（◎○▲△☆＝AI上位5頭）」と買い目候補プールを厳密に連動させる。
    ・堅実重視(solid): 印（◎○▲△☆＝AI上位5頭）のみを買い目対象とし、無印の超大穴を排除する。
    ・標準バランス(balanced): 印（◎○▲△☆）を軸・相手の基本とし、相手はAI上位7頭（＋穴注目）まで。
    ・穴狙い特化(upset): 印（◎○▲△☆）＋穴注目（単勝EV>=1.15かつAI上位9頭以内）を対象とする。
    """
    df = horses_df.copy()
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce").fillna(25.0)
    df["popularity"] = df["win_odds"].rank(method="min", ascending=True).astype(int)
    df["ai_p"] = pd.to_numeric(df["pred_win_pct"], errors="coerce").fillna(1.0) / 100.0
    df["win_ev"] = df["ai_p"] * df["win_odds"]
    df = df.sort_values("ai_p", ascending=False).reset_index(drop=True)
    df["ai_rank"] = np.arange(1, len(df) + 1)

    if exclude_fav1:
        pool = df[df["popularity"] != 1].copy()
        if len(pool) < 5:
            pool = df.copy()
    else:
        pool = df.copy()

    marked_nums = set(pool["horse_num"].astype(int).head(5).tolist())
    top9 = pool.head(9)
    value_nums = set(top9[top9["win_ev"] >= 1.15]["horse_num"].astype(int).tolist())

    if style_mode == "solid":
        allowed_nums = set(marked_nums)
        if len(allowed_nums) < 5:
            allowed_nums |= set(pool["horse_num"].astype(int).head(5).tolist())
    elif style_mode == "upset":
        allowed_nums = set(marked_nums) | value_nums | set(pool["horse_num"].astype(int).head(7).tolist())
    else:
        top7 = pool.head(7)
        bal_extra = set(top7[top7["win_ev"] >= 1.05]["horse_num"].astype(int).tolist())
        allowed_nums = set(marked_nums) | bal_extra
        if len(allowed_nums) < 6:
            allowed_nums |= set(pool["horse_num"].astype(int).head(6).tolist())

    if exclude_fav1:
        fav1_set = set(df[df["popularity"] == 1]["horse_num"].astype(int).tolist())
        if len(allowed_nums - fav1_set) >= 5:
            allowed_nums -= fav1_set
        if len(marked_nums - fav1_set) >= 3:
            marked_nums -= fav1_set

    return pool, marked_nums, allowed_nums


def build_budget_mode_tickets(
    horses_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    exacta_df: pd.DataFrame,
    trio_df: pd.DataFrame,
    exclude_fav1: bool = False,
    style_mode: str = "balanced",
) -> dict:
    """
    300円 / 500円 / 2000円モードの買い目を構築する（03_app.pyで200円モードも自動派生）。
    出走表の「印（◎○▲△☆）」と買い目が一致するよう、印馬中心で買い目を組み立てる。
    """
    pool, marked_nums, allowed_nums = _rank_horses_for_bets(
        horses_df=horses_df,
        exclude_fav1=exclude_fav1,
        style_mode=style_mode,
    )

    pool_nums = pool["horse_num"].astype(int).tolist()
    name_map = dict(zip(horses_df["horse_num"].astype(int), horses_df["horse_name"].astype(str)))
    odds_map = dict(zip(horses_df["horse_num"].astype(int), horses_df["win_odds"].astype(float)))
    prob_map = dict(zip(horses_df["horse_num"].astype(int), horses_df["pred_win_pct"].astype(float)))

    axis_1 = pool_nums[0]

    mark_candidates = [h for h in pool_nums[1:] if h in marked_nums]
    if not mark_candidates:
        mark_candidates = pool_nums[1:5]

    if style_mode == "solid":
        axis_2 = mark_candidates[0]
    elif style_mode == "upset":
        cand_ax2 = [h for h in pool_nums[1:] if h in allowed_nums] or mark_candidates
        axis_2 = max(
            cand_ax2[:5],
            key=lambda h: (prob_map[h] / 100.0) * min(odds_map[h], 40.0),
        )
    else:
        top_marks = mark_candidates[:3]
        axis_2 = max(
            top_marks,
            key=lambda h: ((prob_map[h] / 100.0) ** 0.7) * min(odds_map[h], 30.0),
        )

    axis_list = [axis_1, axis_2]

    rem_marks = [h for h in pool_nums if h in marked_nums and h not in axis_list]
    rem_allowed = [h for h in pool_nums if h in allowed_nums and h not in axis_list and h not in rem_marks]
    rem_others = [h for h in pool_nums if h not in axis_list and h not in rem_marks and h not in rem_allowed]

    if style_mode == "upset":
        rem_allowed.sort(key=lambda h: (prob_map[h] / 100.0) * min(odds_map[h], 50.0), reverse=True)

    followers_list = (rem_marks + rem_allowed + rem_others)[:5]

    # 三連複 2頭軸5頭流し（5点）テーブル構築
    trio_lookup = {}
    if trio_df is not None and not trio_df.empty:
        for _, r in trio_df.iterrows():
            k = tuple(sorted([int(r["horse_num_1"]), int(r["horse_num_2"]), int(r["horse_num_3"])]))
            trio_lookup[k] = r

    trio_5_rows = []
    for f_h in followers_list:
        comb_t = tuple(sorted([axis_1, axis_2, f_h]))
        if comb_t in trio_lookup:
            tr = trio_lookup[comb_t]
            o_t = float(tr["三連複オッズ"])
            p_t = float(tr["三連複的中率(%)"])
            ev_t = float(tr["三連複期待値(EV)"])
        else:
            pa, pb, pc = prob_map[axis_1] / 100.0, prob_map[axis_2] / 100.0, prob_map[f_h] / 100.0
            p_raw = 0.0
            for x, y, z in itertools.permutations((pa, pb, pc)):
                p_raw += x * (y / max(1e-6, 1.0 - x)) * (z / max(1e-6, 1.0 - x - y))
            p_t = round(p_raw * 100.0, 2)
            o_t = round(float(np.clip(0.75 / max(1e-6, p_raw), 3.0, 999.9)), 1)
            ev_t = round((p_t / 100.0) * o_t, 2)

        trio_5_rows.append({
            "券種": "三連複",
            "組み合わせ": f"{comb_t[0]:02d} - {comb_t[1]:02d} - {comb_t[2]:02d}",
            "馬名トリオ": f"{name_map[comb_t[0]]} × {name_map[comb_t[1]]} × {name_map[comb_t[2]]}",
            "三連複オッズ": o_t,
            "三連複的中率(%)": p_t,
            "三連複期待値(EV)": ev_t,
            "購入額(円)": 100,
            "想定払戻(円)": int(round(o_t * 100)),
        })

    df_500_tickets = pd.DataFrame(trio_5_rows)

    # 300円モード（および200円モード元データ）：印馬中心の全券種バランス厳選3点
    if style_mode == "solid":
        p_exp, ev_cap = 1.15, 2.5
    elif style_mode == "upset":
        p_exp, ev_cap = 0.55, 5.0
    else:
        p_exp, ev_cap = 0.85, 3.5

    cand_300 = []

    for h in pool_nums:
        if h not in allowed_nums:
            continue
        p_w = prob_map[h] / 100.0
        o_w = odds_map[h]
        ev_w = p_w * o_w
        if ev_w < 0.75:
            continue
        score = (p_w ** p_exp) * min(ev_w, ev_cap)
        if h == axis_1:
            score *= 1.15
        cand_300.append({
            "券種": "単勝",
            "組み合わせ": f"{h:02d}",
            "対象馬名": name_map[h],
            "オッズ": round(o_w, 1),
            "的中率(%)": round(p_w * 100.0, 2),
            "期待値(EV)": round(ev_w, 2),
            "購入額(円)": 100,
            "想定払戻(円)": int(round(o_w * 100)),
            "_score": score,
        })

    if pairs_df is not None and not pairs_df.empty:
        for _, r in pairs_df.iterrows():
            h1, h2 = int(r["horse_num_1"]), int(r["horse_num_2"])
            if h1 not in allowed_nums or h2 not in allowed_nums:
                continue
            if style_mode != "upset" and (h1 not in marked_nums and h2 not in marked_nums):
                continue

            p_u = float(r["馬連的中率(%)"]) / 100.0
            o_u = float(r["馬連オッズ"])
            ev_u = float(r["馬連期待値(EV)"])
            if p_u >= 0.015 and ev_u >= 0.85:
                sc_u = (p_u ** p_exp) * min(ev_u, ev_cap)
                if axis_1 in (h1, h2):
                    sc_u *= 1.20
                cand_300.append({
                    "券種": "馬連",
                    "組み合わせ": str(r["組み合わせ"]),
                    "対象馬名": str(r["馬名ペア"]),
                    "オッズ": o_u,
                    "的中率(%)": float(r["馬連的中率(%)"]),
                    "期待値(EV)": ev_u,
                    "購入額(円)": 100,
                    "想定払戻(円)": int(round(o_u * 100)),
                    "_score": sc_u,
                })

            p_wd = float(r["ワイド的中率(%)"]) / 100.0
            o_wd = float(r["ワイドオッズ"])
            ev_wd = float(r["ワイド期待値(EV)"])
            if p_wd >= 0.04 and ev_wd >= 0.85:
                sc_wd = (p_wd ** p_exp) * min(ev_wd, ev_cap)
                if axis_1 in (h1, h2):
                    sc_wd *= 1.20
                cand_300.append({
                    "券種": "ワイド",
                    "組み合わせ": str(r["組み合わせ"]),
                    "対象馬名": str(r["馬名ペア"]),
                    "オッズ": o_wd,
                    "的中率(%)": float(r["ワイド的中率(%)"]),
                    "期待値(EV)": ev_wd,
                    "購入額(円)": 100,
                    "想定払戻(円)": int(round(o_wd * 100)),
                    "_score": sc_wd,
                })

    if trio_df is not None and not trio_df.empty:
        for _, r in trio_df.iterrows():
            h1, h2, h3 = int(r["horse_num_1"]), int(r["horse_num_2"]), int(r["horse_num_3"])
            if h1 not in allowed_nums or h2 not in allowed_nums or h3 not in allowed_nums:
                continue
            mark_cnt = sum(1 for x in (h1, h2, h3) if x in marked_nums)
            if mark_cnt < 2:
                continue
            p_tr = float(r["三連複的中率(%)"]) / 100.0
            o_tr = float(r["三連複オッズ"])
            ev_tr = float(r["三連複期待値(EV)"])
            if p_tr >= 0.008 and ev_tr >= 0.90:
                sc_tr = (p_tr ** p_exp) * min(ev_tr, ev_cap)
                if axis_1 in (h1, h2, h3):
                    sc_tr *= 1.15
                cand_300.append({
                    "券種": "三連複",
                    "組み合わせ": str(r["組み合わせ"]),
                    "対象馬名": str(r["馬名トリオ"]),
                    "オッズ": o_tr,
                    "的中率(%)": float(r["三連複的中率(%)"]),
                    "期待値(EV)": ev_tr,
                    "購入額(円)": 100,
                    "想定払戻(円)": int(round(o_tr * 100)),
                    "_score": sc_tr,
                })

    if cand_300:
        c300_df = pd.DataFrame(cand_300).sort_values("_score", ascending=False)
        picked_300 = c300_df.head(3).drop(columns=["_score"]).reset_index(drop=True)
    else:
        fb_rows = []
        top_m = pool_nums[:4]
        for h1, h2 in itertools.combinations(top_m, 2):
            sub = pairs_df[(pairs_df["horse_num_1"] == min(h1, h2)) & (pairs_df["horse_num_2"] == max(h1, h2))]
            if not sub.empty:
                r = sub.iloc[0]
                fb_rows.append({
                    "券種": "ワイド",
                    "組み合わせ": str(r["組み合わせ"]),
                    "対象馬名": str(r["馬名ペア"]),
                    "オッズ": float(r["ワイドオッズ"]),
                    "的中率(%)": float(r["ワイド的中率(%)"]),
                    "期待値(EV)": float(r["ワイド期待値(EV)"]),
                    "購入額(円)": 100,
                    "想定払戻(円)": int(round(float(r["ワイドオッズ"]) * 100)),
                })
            if len(fb_rows) >= 3:
                break
        picked_300 = pd.DataFrame(fb_rows)

    # 2000円モード（三連複5点 + 馬単4頭BOX12点 + ワイド3頭BOX3点 = 20点・2000円）
    box4_pool = [h for h in pool_nums if h in allowed_nums][:4]
    if len(box4_pool) < 4:
        box4_pool = pool_nums[:4]
    exacta_box_nums = sorted(box4_pool)
    wide_box_nums = sorted(box4_pool[:3])

    ex_lookup = {}
    if exacta_df is not None and not exacta_df.empty:
        for _, r in exacta_df.iterrows():
            ex_lookup[(int(r["horse_num_1"]), int(r["horse_num_2"]))] = r

    exacta_12_rows = []
    for h1, h2 in itertools.permutations(exacta_box_nums, 2):
        if (h1, h2) in ex_lookup:
            er = ex_lookup[(h1, h2)]
            o_e = float(er["馬単オッズ"])
            p_e = float(er["馬単的中率(%)"])
            ev_e = float(er["馬単期待値(EV)"])
        else:
            p1, p2 = prob_map[h1] / 100.0, prob_map[h2] / 100.0
            pe_raw = p1 * (p2 / max(1e-6, 1.0 - p1))
            p_e = round(pe_raw * 100.0, 2)
            o_e = round(float(np.clip(0.75 / max(1e-6, pe_raw), 2.0, 1999.9)), 1)
            ev_e = round((p_e / 100.0) * o_e, 2)
        exacta_12_rows.append({
            "券種": "馬単",
            "組み合わせ": f"{h1:02d} → {h2:02d}",
            "馬名順列": f"{name_map[h1]} → {name_map[h2]}",
            "馬単オッズ": o_e,
            "馬単的中率(%)": p_e,
            "馬単期待値(EV)": ev_e,
            "購入額(円)": 100,
            "想定払戻(円)": int(round(o_e * 100)),
        })
    df_exacta_2000 = pd.DataFrame(exacta_12_rows).sort_values("馬単的中率(%)", ascending=False).reset_index(drop=True)

    pair_lookup = {}
    if pairs_df is not None and not pairs_df.empty:
        for _, r in pairs_df.iterrows():
            pair_lookup[(int(r["horse_num_1"]), int(r["horse_num_2"]))] = r

    wide_3_rows = []
    for h1, h2 in itertools.combinations(wide_box_nums, 2):
        k = (min(h1, h2), max(h1, h2))
        if k in pair_lookup:
            pr = pair_lookup[k]
            o_w = float(pr["ワイドオッズ"])
            p_w = float(pr["ワイド的中率(%)"])
            ev_w = float(pr["ワイド期待値(EV)"])
        else:
            o_w, p_w, ev_w = 3.5, 25.0, 0.88
        wide_3_rows.append({
            "券種": "ワイド",
            "組み合わせ": f"{k[0]:02d} - {k[1]:02d}",
            "馬名ペア": f"{name_map[k[0]]} × {name_map[k[1]]}",
            "ワイドオッズ": o_w,
            "ワイド的中率(%)": p_w,
            "ワイド期待値(EV)": ev_w,
            "購入額(円)": 100,
            "想定払戻(円)": int(round(o_w * 100)),
        })
    df_wide_2000 = pd.DataFrame(wide_3_rows)

    return {
        "mode_300": {
            "tickets_df": picked_300,
            "total_cost": int(picked_300["購入額(円)"].sum()) if not picked_300.empty else 300,
        },
        "mode_500": {
            "axis": axis_list,
            "followers": followers_list,
            "tickets_df": df_500_tickets,
            "total_cost": int(df_500_tickets["購入額(円)"].sum()) if not df_500_tickets.empty else 500,
        },
        "mode_2000": {
            "trio_axis": axis_list,
            "trio_followers": followers_list,
            "exacta_box": exacta_box_nums,
            "wide_box": wide_box_nums,
            "trio_df": df_500_tickets.copy(),
            "exacta_df": df_exacta_2000,
            "wide_df": df_wide_2000,
            "total_cost": 2000,
        },
    }


def build_trio_strategies(
    trio_df: pd.DataFrame,
    horses_df: pd.DataFrame,
    ev_threshold: float = 1.15,
    min_prob_pct: float = 0.4,
    max_single_tickets: int = 10,
    max_followers: int = 8,
    exclude_fav1: bool = False,
) -> dict:
    """詳細タブの三連複フォーメーションを構築する"""
    df = horses_df.copy()
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce").fillna(25.0)
    df["popularity"] = df["win_odds"].rank(method="min", ascending=True).astype(int)
    df = df.sort_values("pred_win_pct", ascending=False).reset_index(drop=True)
    if exclude_fav1:
        sub = df[df["popularity"] != 1]
        if len(sub) >= 6:
            df = sub.reset_index(drop=True)

    nums = df["horse_num"].astype(int).tolist()
    col1 = nums[:2]
    col2 = nums[:4]
    col3 = nums[:min(len(nums), max(5, max_followers))]

    valid_combos = set()
    for a in col1:
        for b in col2:
            for c in col3:
                if len({a, b, c}) == 3:
                    valid_combos.add(tuple(sorted([a, b, c])))

    if trio_df is not None and not trio_df.empty and valid_combos:
        mask = [
            tuple(sorted([int(r["horse_num_1"]), int(r["horse_num_2"]), int(r["horse_num_3"])])) in valid_combos
            for _, r in trio_df.iterrows()
        ]
        form_df = trio_df[mask].sort_values("三連複期待値(EV)", ascending=False).reset_index(drop=True)
        single_df = (
            trio_df[(trio_df["三連複期待値(EV)"] >= ev_threshold) & (trio_df["三連複的中率(%)"] >= min_prob_pct)]
            .sort_values("三連複期待値(EV)", ascending=False)
            .head(max_single_tickets)
            .reset_index(drop=True)
        )
    else:
        form_df = pd.DataFrame()
        single_df = pd.DataFrame()

    return {
        "formation": {
            "col1": col1,
            "col2": col2,
            "col3": col3,
            "tickets_df": form_df,
        },
        "single": single_df,
    }