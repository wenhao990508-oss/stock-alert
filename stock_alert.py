"""
台股 K棒 + 成交量 + 布林通道 自動提醒
依圖中規則判斷買進 / 出場訊號，透過 ntfy 推播到手機。
環境變數：
  STOCKS      監控清單，例 "6770:力積電,2330:台積電"
  NTFY_TOPIC  ntfy 主題名稱（手機 App 訂閱同一個名稱）
  FORCE=1     忽略「今天是否有新K棒」檢查（測試用）
  TEST=1      直接送一則測試通知
"""
import os
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

# ===== 參數（對應圖片規則，可自行調整）=====
BB_N, BB_K = 20, 2          # 布林通道 20 日、2 倍標準差
VOL_N = 10                  # 10 日均量
BREAKOUT_VOL = 1.3          # 突破確認：量 >= 均量 × 1.3
BLOWOFF_VOL = 2.0           # 爆量：量 > 均量 × 2
NEAR_MID = 0.01             # 回踩中軌：最低價距中軌 1% 內
NEAR_UP = 0.01              # 接近上軌：最高價距上軌 1% 內

TZ = ZoneInfo("Asia/Taipei")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "").strip()


def parse_stocks():
    raw = os.getenv("STOCKS", "6770:力積電")
    out = []
    for item in raw.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        code, _, name = item.partition(":")
        out.append((code.strip(), name.strip() or code.strip()))
    return out


def fetch(code):
    """上市試 .TW，上櫃試 .TWO"""
    for suffix in (".TW", ".TWO"):
        df = yf.download(code + suffix, period="6mo", interval="1d",
                         progress=False, auto_adjust=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = df[df["Volume"] > 0]
            if len(df) > BB_N + 2:
                return df
    return None


def add_indicators(df):
    c = df["Close"]
    df["MID"] = c.rolling(BB_N).mean()
    std = c.rolling(BB_N).std(ddof=0)
    df["UP"] = df["MID"] + BB_K * std
    df["DN"] = df["MID"] - BB_K * std
    df["VMA"] = df["Volume"].rolling(VOL_N).mean()
    return df


def check(df):
    """回傳 [(類型, 標題, 說明)]，類型 buy / sell"""
    t, p, pp = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    o, h, l, c, v = t.Open, t.High, t.Low, t.Close, t.Volume
    red, black = c > o, c < o
    vr = v / t.VMA
    body = abs(c - o)
    rng = h - l
    upper_shadow = h - max(o, c)
    sig = []

    # ── 上漲 / 進場 ──
    if red and c > p.High and vr >= BREAKOUT_VOL and c > t.UP and t.MID > p.MID:
        sig.append(("buy", "🟢 A級 突破型買點",
                    f"紅K突破前高、收上布林上軌、量為均量{vr:.2f}倍、中軌向上"))

    if red and l <= t.MID * (1 + NEAR_MID) and c > t.MID and v < t.VMA:
        sig.append(("buy", "🟢 B級 回踩中軌買點",
                    f"回到中軌附近量縮({vr:.2f}倍)，紅K重新站回中軌"))

    # ── 下跌 / 出場 ──
    if black and c < t.MID and p.Close >= p.MID and vr > 1.0:
        sig.append(("sell", "🔴 C級 跌破中軌",
                    f"黑K跌破中軌且量增({vr:.2f}倍)，考慮減碼；明天站不回就出場"))

    if p.Close < p.MID and pp.Close >= pp.MID and c < t.MID:
        sig.append(("sell", "🔴 C級 無法站回中軌",
                    "昨天跌破中軌、今天仍站不回 → 剩餘部位出場"))

    long_upper = rng > 0 and upper_shadow >= max(body, rng * 0.4)
    long_red = red and rng > 0 and body >= rng * 0.6 and body / p.Close >= 0.03
    if (long_upper or long_red) and vr > BLOWOFF_VOL and h >= t.UP * (1 - NEAR_UP):
        kind = "長上影線" if long_upper else "長紅K"
        sig.append(("sell", "🟠 D級 爆量" + kind,
                    f"接近上軌爆量({vr:.2f}倍)，注意賣壓，可先賣 1/3~1/2"))

    if p.Close > p.UP and black and c < t.UP and vr > 1.0:
        sig.append(("sell", "🟣 E級 假突破",
                    f"昨天收上軌、今天黑K跌回上軌下方且放量({vr:.2f}倍)"))

    info = (f"收盤 {c:.2f}｜量比 {vr:.2f}\n"
            f"上軌 {t.UP:.2f}｜中軌 {t.MID:.2f}｜下軌 {t.DN:.2f}")
    return sig, info


def notify(title, message, priority=3, tags=None):
    print(f"\n[{title}]\n{message}")
    if not NTFY_TOPIC:
        return
    r = requests.post("https://ntfy.sh/", json={
        "topic": NTFY_TOPIC, "title": title, "message": message,
        "priority": priority, "tags": tags or []}, timeout=15)
    r.raise_for_status()


def main():
    if os.getenv("TEST") == "1":
        notify("✅ 股票提醒測試", "收到這則就代表設定成功！", 4)
        return

    today = dt.datetime.now(TZ).date()
    force = os.getenv("FORCE") == "1"

    for code, name in parse_stocks():
        df = fetch(code)
        if df is None:
            print(f"{code} 抓不到資料，略過")
            continue
        last_date = df.index[-1].date()
        if not force and last_date != today:
            print(f"{code} 最新K棒 {last_date} 不是今天（休市或資料未更新），略過")
            continue

        sigs, info = check(add_indicators(df))
        if not sigs:
            print(f"{name}({code}) {last_date} 無訊號\n{info}")
            continue
        for kind, title, desc in sigs:
            notify(f"{name}({code}) {title}",
                   f"{desc}\n{info}\n日期 {last_date}",
                   priority=5 if kind == "sell" else 4,
                   tags=["chart_with_downwards_trend" if kind == "sell"
                         else "chart_with_upwards_trend"])


if __name__ == "__main__":
    main()
