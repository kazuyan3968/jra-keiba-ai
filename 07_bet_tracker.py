from datetime import datetime
import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

HISTORY_PATH = "data/bet_history.csv"

HISTORY_COLS = [
    "bet_id",
    "saved_at",
    "race_id",
    "date",
    "venue",
    "race_num",
    "race_title",
    "mode",
    "ticket_type",
    "combination",
    "horse_names",
    "amount",
    "pred_odds",
    "status",
    "payout",
    "actual_result",
]

VENUE_TO_IPATGO = {
    "札幌": "SAPPORO",
    "函館": "HAKODATE",
    "福島": "FUKUSHIMA",
    "新潟": "NIIGATA",
    "東京": "TOKYO",
    "中山": "NAKAYAMA",
    "中京": "CHUKYO",
    "京都": "KYOTO",
    "阪神": "HANSHIN",
    "小倉": "KOKURA",
}

TICKET_TO_IPATGO = {
    "単勝": "TANSYO",
    "複勝": "FUKUSYO",
    "馬連": "UMAREN",
    "ワイド": "WIDE",
    "馬単": "UMATAN",
    "三連複": "SANRENPUKU",
    "三連単": "SANRENTAN",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/",
}


def decode_html(resp: requests.Response) -> str:
    raw = resp.content
    head = raw[:3000].lower()
    if b"utf-8" in head or b"utf8" in head:
        return raw.decode("utf-8", errors="replace")
    if b"euc-jp" in head or b"euc_jp" in head:
        return raw.decode("euc_jp", errors="replace")
    if b"shift_jis" in head or b"shift-jis" in head or b"cp932" in head:
        return raw.decode("cp932", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("euc_jp", errors="replace")


def normalize_ticket_type(raw_type: str) -> str:
    s = str(raw_type)
    for base in ("三連単", "三連複", "3連複", "馬単", "馬連", "ワイド", "単勝", "複勝"):
        if base in s:
            return "三連複" if base == "3連複" else base
    return s


def normalize_combination(ticket_type: str, combo_str: str) -> str:
    """馬番組み合わせを照合用の標準形式（例: 01-05-14, 05>01, 05）に統一する"""
    nums = [int(n) for n in re.findall(r"\d+", str(combo_str))]
    if not nums:
        return str(combo_str).strip()
    base = normalize_ticket_type(ticket_type)
    if base in ("単勝", "複勝"):
        return f"{nums[0]:02d}"
    if base in ("馬単", "三連単"):
        return ">".join(f"{n:02d}" for n in nums)
    return "-".join(f"{n:02d}" for n in sorted(nums))


def load_bet_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=HISTORY_COLS)
    df = pd.read_csv(
        HISTORY_PATH,
        dtype={"bet_id": str, "race_id": str, "combination": str, "status": str, "actual_result": str},
    )
    for col in HISTORY_COLS:
        if col not in df.columns:
            df[col] = ""
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(100).astype(int)
    df["payout"] = pd.to_numeric(df["payout"], errors="coerce").fillna(0).astype(int)
    df["pred_odds"] = pd.to_numeric(df["pred_odds"], errors="coerce").fillna(0.0)
    return df[HISTORY_COLS]


def save_bet_history(df: pd.DataFrame):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    df[HISTORY_COLS].to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


def extract_mode_ticket_rows(current_mode: str, budget_modes: dict) -> list[dict]:
    """現在選択中のモード（300 / 500 / 2000）から標準化した買い目リストを作成する"""
    rows = []
    if current_mode == "300":
        df = budget_modes["mode_300"]["tickets_df"]
        for _, r in df.iterrows():
            t_type = normalize_ticket_type(r["券種"])
            rows.append({
                "mode": "300円",
                "ticket_type": t_type,
                "combination": normalize_combination(t_type, r["組み合わせ"]),
                "horse_names": str(r["対象馬名"]),
                "amount": int(r["購入額(円)"]),
                "pred_odds": float(r["オッズ"]),
            })
    elif current_mode == "500":
        df = budget_modes["mode_500"]["tickets_df"]
        for _, r in df.iterrows():
            rows.append({
                "mode": "500円",
                "ticket_type": "三連複",
                "combination": normalize_combination("三連複", r["組み合わせ"]),
                "horse_names": str(r["馬名トリオ"]),
                "amount": int(r["購入額(円)"]),
                "pred_odds": float(r["三連複オッズ"]),
            })
    else:
        m2000 = budget_modes["mode_2000"]
        for _, r in m2000["trio_df"].iterrows():
            rows.append({
                "mode": "2000円",
                "ticket_type": "三連複",
                "combination": normalize_combination("三連複", r["組み合わせ"]),
                "horse_names": str(r["馬名トリオ"]),
                "amount": 100,
                "pred_odds": float(r["三連複オッズ"]),
            })
        for _, r in m2000["exacta_df"].iterrows():
            rows.append({
                "mode": "2000円",
                "ticket_type": "馬単",
                "combination": normalize_combination("馬単", r["組み合わせ"]),
                "horse_names": str(r["馬名順列"]),
                "amount": 100,
                "pred_odds": float(r["馬単オッズ"]),
            })
        for _, r in m2000["wide_df"].iterrows():
            rows.append({
                "mode": "2000円",
                "ticket_type": "ワイド",
                "combination": normalize_combination("ワイド", r["組み合わせ"]),
                "horse_names": str(r["馬名ペア"]),
                "amount": 100,
                "pred_odds": float(r["ワイドオッズ"]),
            })
    return rows


def append_bets_to_history(race_meta: dict, ticket_rows: list[dict]) -> tuple[int, int]:
    """買い目を履歴CSVに追記する。重複する同一レース・同一モード・同一買い目はスキップ"""
    hist_df = load_bet_history()
    existing_keys = set(
        zip(
            hist_df["race_id"].astype(str),
            hist_df["mode"].astype(str),
            hist_df["ticket_type"].astype(str),
            hist_df["combination"].astype(str),
        )
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_records = []
    skipped = 0

    for idx, t in enumerate(ticket_rows):
        key = (str(race_meta["race_id"]), t["mode"], t["ticket_type"], t["combination"])
        if key in existing_keys:
            skipped += 1
            continue
        existing_keys.add(key)
        new_records.append({
            "bet_id": f"{race_meta['race_id']}_{t['mode']}_{idx}_{int(time.time())}",
            "saved_at": now_str,
            "race_id": str(race_meta["race_id"]),
            "date": str(race_meta["date"]),
            "venue": str(race_meta["venue"]),
            "race_num": int(race_meta["race_num"]),
            "race_title": str(race_meta["race_title"]),
            "mode": t["mode"],
            "ticket_type": t["ticket_type"],
            "combination": t["combination"],
            "horse_names": t["horse_names"],
            "amount": int(t["amount"]),
            "pred_odds": round(float(t["pred_odds"]), 1),
            "status": "未確定",
            "payout": 0,
            "actual_result": "",
        })

    if new_records:
        updated_df = pd.concat([hist_df, pd.DataFrame(new_records)], ignore_index=True)
        save_bet_history(updated_df)

    return len(new_records), skipped


def _extract_payout_ints(td_el) -> list[int]:
    """払戻金セルから改行区切りで複数の金額（100円あたり払戻）を抽出する"""
    if td_el is None:
        return []
    for br in td_el.find_all("br"):
        br.replace_with("\n")
    text = td_el.get_text("\n")
    vals = []
    for line in text.splitlines():
        cleaned = line.replace(",", "").replace("円", "").strip()
        m = re.search(r"(\d+)", cleaned)
        if m:
            vals.append(int(m.group(1)))
    return vals


def fetch_official_payouts(race_id: str) -> dict | None:
    """レース結果ページから確定着順（1〜3着）と券種別公式払戻金辞書を取得する"""
    # 1. race.netkeiba.com 結果ページ
    try:
        time.sleep(1.2)
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(decode_html(resp), "lxml")
            payout_table = soup.select("table.Payout_Detail_Table")
            if payout_table:
                payouts: dict[str, dict[str, int]] = {
                    "単勝": {},
                    "複勝": {},
                    "馬連": {},
                    "ワイド": {},
                    "馬単": {},
                    "三連複": {},
                    "三連単": {},
                }
                cls_map = {
                    "Tansho": "単勝",
                    "Fukusho": "複勝",
                    "Umaren": "馬連",
                    "Wide": "ワイド",
                    "Umatan": "馬単",
                    "Fuku3": "三連複",
                    "Tan3": "三連単",
                }
                for cls_name, jp_name in cls_map.items():
                    tr = soup.select_one(f"table.Payout_Detail_Table tr.{cls_name}")
                    if not tr:
                        continue
                    res_td = tr.select_one("td.Result")
                    pay_td = tr.select_one("td.Payout")
                    pay_vals = _extract_payout_ints(pay_td)

                    combos = []
                    if res_td:
                        uls = res_td.find_all("ul")
                        if uls:
                            for ul in uls:
                                nums = [int(x) for x in re.findall(r"\d+", ul.get_text(" ", strip=True))]
                                if nums:
                                    combos.append(nums)
                        else:
                            divs = res_td.find_all("div")
                            for dv in divs:
                                nums = [int(x) for x in re.findall(r"\d+", dv.get_text(" ", strip=True))]
                                if nums:
                                    combos.append(nums)

                    for nums, pay in zip(combos, pay_vals):
                        if jp_name in ("単勝", "複勝"):
                            key = f"{nums[0]:02d}"
                        elif jp_name in ("馬単", "三連単"):
                            key = ">".join(f"{n:02d}" for n in nums)
                        else:
                            key = "-".join(f"{n:02d}" for n in sorted(nums))
                        payouts[jp_name][key] = pay

                top3 = []
                for tr in soup.select("table#All_Result_Table tr.HorseList")[:3]:
                    rank_el = tr.select_one("td.Result_Num div.Rank")
                    num_el = tr.select_one("td.Num.Txt_C div") or tr.select_one("td:nth-of-type(3)")
                    if rank_el and num_el:
                        r_txt = rank_el.get_text(strip=True)
                        u_txt = num_el.get_text(strip=True)
                        if r_txt.isdigit() and u_txt.isdigit():
                            top3.append(f"{r_txt}着:{int(u_txt):02d}")

                if any(len(v) > 0 for v in payouts.values()):
                    return {"payouts": payouts, "top3_text": " / ".join(top3)}
    except Exception:
        pass

    # 2. db.netkeiba.com 結果ページ（フォールバック）
    try:
        time.sleep(1.5)
        db_url = f"https://db.netkeiba.com/race/{race_id}/"
        resp = requests.get(db_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(decode_html(resp), "lxml")
            pay_tables = soup.select("table.pay_table_01")
            if pay_tables:
                payouts = {
                    "単勝": {},
                    "複勝": {},
                    "馬連": {},
                    "ワイド": {},
                    "馬単": {},
                    "三連複": {},
                    "三連単": {},
                }
                for tbl in pay_tables:
                    for tr in tbl.find_all("tr"):
                        th = tr.find("th")
                        tds = tr.find_all("td")
                        if not th or len(tds) < 2:
                            continue
                        jp_name = th.get_text(strip=True)
                        if jp_name not in payouts:
                            continue

                        res_td = tds[0]
                        for br in res_td.find_all("br"):
                            br.replace_with("\n")
                        combo_lines = [ln.strip() for ln in res_td.get_text("\n").splitlines() if ln.strip()]
                        pay_vals = _extract_payout_ints(tds[1])

                        for ln, pay in zip(combo_lines, pay_vals):
                            nums = [int(x) for x in re.findall(r"\d+", ln)]
                            if not nums:
                                continue
                            if jp_name in ("単勝", "複勝"):
                                key = f"{nums[0]:02d}"
                            elif jp_name in ("馬単", "三連単"):
                                key = ">".join(f"{n:02d}" for n in nums)
                            else:
                                key = "-".join(f"{n:02d}" for n in sorted(nums))
                            payouts[jp_name][key] = pay

                top3 = []
                res_tbl = soup.find("table", class_="race_table_01")
                if res_tbl:
                    for tr in res_tbl.find_all("tr")[1:4]:
                        tds = tr.find_all("td")
                        if len(tds) >= 3 and tds[0].get_text(strip=True).isdigit() and tds[2].get_text(strip=True).isdigit():
                            top3.append(f"{tds[0].get_text(strip=True)}着:{int(tds[2].get_text(strip=True)):02d}")

                if any(len(v) > 0 for v in payouts.values()):
                    return {"payouts": payouts, "top3_text": " / ".join(top3)}
    except Exception:
        pass

    return None


def settle_pending_bets() -> dict:
    """未確定の保存済み買い目をWeb結果と照合し、的中判定・払戻金を自動更新する"""
    hist_df = load_bet_history()
    if hist_df.empty:
        return {"checked_races": 0, "settled_tickets": 0, "hit_tickets": 0}

    pending_mask = hist_df["status"] == "未確定"
    pending_race_ids = hist_df.loc[pending_mask, "race_id"].unique().tolist()

    checked_races = 0
    settled_tickets = 0
    hit_tickets = 0

    for rid in pending_race_ids:
        result_data = fetch_official_payouts(str(rid))
        if result_data is None:
            continue

        checked_races += 1
        payouts = result_data["payouts"]
        top3_text = result_data["top3_text"]

        idx_list = hist_df.index[(hist_df["race_id"] == rid) & (hist_df["status"] == "未確定")].tolist()
        for idx in idx_list:
            t_type = normalize_ticket_type(hist_df.at[idx, "ticket_type"])
            combo = normalize_combination(t_type, hist_df.at[idx, "combination"])
            amount = int(hist_df.at[idx, "amount"])

            pay_per_100 = payouts.get(t_type, {}).get(combo, 0)
            if pay_per_100 > 0:
                hist_df.at[idx, "status"] = "的中"
                hist_df.at[idx, "payout"] = int((pay_per_100 * amount) // 100)
                hit_tickets += 1
            else:
                hist_df.at[idx, "status"] = "不的中"
                hist_df.at[idx, "payout"] = 0

            hist_df.at[idx, "actual_result"] = top3_text
            settled_tickets += 1

    if settled_tickets > 0:
        save_bet_history(hist_df)

    return {
        "checked_races": checked_races,
        "settled_tickets": settled_tickets,
        "hit_tickets": hit_tickets,
    }


def delete_race_from_history(race_id: str, mode: str | None = None):
    hist_df = load_bet_history()
    if hist_df.empty:
        return
    if mode:
        mask = ~((hist_df["race_id"] == str(race_id)) & (hist_df["mode"] == str(mode)))
    else:
        mask = hist_df["race_id"] != str(race_id)
    save_bet_history(hist_df[mask].reset_index(drop=True))


def clear_all_history():
    save_bet_history(pd.DataFrame(columns=HISTORY_COLS))


def build_ipat_exports(race_meta: dict, current_mode: str, budget_modes: dict, ticket_rows: list[dict]) -> dict:
    """即PAT手動入力用の整形テキストと、IPAT補助ツール（ipatgo等）連携用CSVを生成する"""
    venue_jp = str(race_meta["venue"])
    race_num = int(race_meta["race_num"])
    date_ymd = str(race_meta["date"]).replace("-", "")
    venue_en = VENUE_TO_IPATGO.get(venue_jp, "TOKYO")

    lines = [
        f"【JRA 即PAT 投票メモ】 {race_meta['date']} {venue_jp}{race_num}R {race_meta['race_title']}（{current_mode}円モード）",
        "-" * 58,
    ]
    if current_mode == "500":
        m500 = budget_modes["mode_500"]
        ax = "-".join(f"{n:02d}" for n in m500["axis"])
        fl = ",".join(f"{n:02d}" for n in m500["followers"])
        lines.append("券種: 三連複 / 方式: 軸2頭ながし")
        lines.append(f"  軸馬: {ax}")
        lines.append(f"  相手: {fl}")
        lines.append("  金額: 各100円（計5点 / 500円）")
    elif current_mode == "2000":
        m2000 = budget_modes["mode_2000"]
        ax = "-".join(f"{n:02d}" for n in m2000["trio_axis"])
        fl = ",".join(f"{n:02d}" for n in m2000["trio_followers"])
        bx4 = ",".join(f"{n:02d}" for n in m2000["exacta_box"])
        bx3 = ",".join(f"{n:02d}" for n in m2000["wide_box"])
        lines.append(f"1) 三連複 軸2頭ながし | 軸: {ax} / 相手: {fl} | 各100円（5点・500円）")
        lines.append(f"2) 馬単 ボックス       | 選択馬: {bx4} | 各100円（12点・1,200円）")
        lines.append(f"3) ワイド ボックス     | 選択馬: {bx3} | 各100円（3点・300円）")
        lines.append("合計: 20点 / 2,000円")
    else:
        lines.append("券種: 通常投票（期待値トップ3厳選）")
        for t in ticket_rows:
            disp_combo = t["combination"].replace(">", "→")
            lines.append(f"  {t['ticket_type']:4s} | {disp_combo:10s} | {t['amount']}円  ({t['horse_names']})")
        lines.append(f"合計: {len(ticket_rows)}点 / {sum(t['amount'] for t in ticket_rows)}円")

    lines.append("-" * 58)
    lines.append("詳細買い目一覧（1点ごと）:")
    for t in ticket_rows:
        disp_combo = t["combination"].replace(">", "→")
        lines.append(f"  {t['ticket_type']:4s}  {disp_combo:10s}  {t['amount']:4d}円")

    human_text = "\n".join(lines)

    csv_lines = []
    for t in ticket_rows:
        ipat_type = TICKET_TO_IPATGO.get(t["ticket_type"], "TANSYO")
        ipat_combo = t["combination"].replace(">", "-")
        csv_lines.append(f"{date_ymd},{venue_en},{race_num},{ipat_type},NORMAL,N,{ipat_combo},{t['amount']}")

    ipat_csv = "\n".join(csv_lines) + "\n"
    return {
        "human_text": human_text,
        "ipat_csv": ipat_csv,
        "filename": f"ipat_{date_ymd}_{venue_en}_{race_num:02d}R_{current_mode}yen.csv",
    }