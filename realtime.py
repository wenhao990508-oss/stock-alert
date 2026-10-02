"""
盤中即時爆量偵測：成交量突然放大時，用「內外盤」+「價格方向」+「布林位置」
判斷偏買或偏賣，並推播到手機（ntfy）。
資料來源：證交所 MIS 即時行情（約 5 秒更新，上市/上櫃皆可）
環境變數：同 stock_alert.py（STOCKS、NTFY_TOPIC），另可設 END=13:30
"""
import os
import time
import datetime as dt
from collections import deque

import numpy as np
import requests

from stock_alert import TZ, parse_stocks, fetch, add_indicators, notify

# ===== 參數 =====
POLL_SEC = 10          # 每 10 秒抓一次
WINDOW_SEC = 60        # 爆量偵測視窗：最近 1 分鐘
BASE_MIN = 20          # 比較基準：前 20 分鐘的平均每分鐘量
SURGE_X = 3.0          # 1 分鐘量 >= 基準 × 3 視為「突然爆量」
MIN_LOTS = 200         # 且至少 200 張（避免冷門股小量誤報）
BUY_RATIO = 0.60       # 外盤比 >= 60% 偏買
SELL_RATIO = 0.40      # 外盤比 <= 40% 偏賣
COOLDOWN_MIN = 10      # 同方向 10 分鐘內不重複提醒
SKIP_OPEN_MIN = 5      # 開盤前 5 分鐘量本來就大，不判斷
TRADING_MIN = 270      # 09:00~13:30

MIS = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}


def fnum(x, default=None):
    try:
        v = float(str(x).split("_")[0])
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def mis_query(keys):
    r = requests.get(MIS, params={"ex_ch": "|".join(keys), "json": "1", "delay": "0"},
                     headers=HEADERS, timeout=10)
    r.raise_for_status()
    return [m for m in r.json().get("msgArray", []) if m.get("c")]


def resolve_keys(codes):
    """判斷上市(tse)或上櫃(otc)"""
    keys = [f"{ex}_{c}.tw" for c in codes for ex in ("tse", "otc")]
    found = {m["c"]: f'{m["ex"]}_{m["c"]}.tw' for m in mis_query(keys)}
    return found


class Detector:
    def __init__(self, code, name, daily, today):
        self.code, self.name = code, name
        d = daily[daily.index.date < today]          # 只用昨天以前的完整日K
        self.prev_closes = d["Close"].values[-19:]
        self.prev_mid = d["MID"].iloc[-1]
        self.prev_up = d["UP"].iloc[-1]
        self.prev_close = d["Close"].iloc[-1]
        self.vma_lots = d["Volume"].iloc[-10:].mean() / 1000   # 10日均量（張）
        self.events = deque()        # (時間, 量, 外盤量, 內盤量, 價)
        self.last_v = None
        self.last_bid = self.last_ask = None
        self.price = None
        self.last_alert = {}

    def bands(self, price):
        arr = np.append(self.prev_closes, price)
        mid, std = arr.mean(), arr.std()
        return mid, mid + 2 * std, mid - 2 * std

    def update(self, m, now):
        price = fnum(m.get("z")) or fnum(m.get("pz")) or self.price
        v = fnum(m.get("v"), 0)
        ask, bid = fnum(m.get("a")), fnum(m.get("b"))
        alerts = []
        if price and self.last_v is not None and v > self.last_v:
            dv = v - self.last_v
            if self.last_ask and price >= self.last_ask:
                out, inn = dv, 0                     # 成交在賣價 → 外盤（主動買）
            elif self.last_bid and price <= self.last_bid:
                out, inn = 0, dv                     # 成交在買價 → 內盤（主動賣）
            else:
                out = inn = dv / 2
            self.events.append((now, dv, out, inn, price))
        if price:
            self.price = price
        self.last_v, self.last_ask, self.last_bid = v, ask, bid
        self.high, self.open_ = fnum(m.get("h"), price), fnum(m.get("o"), price)

        # 清掉太舊的資料
        while self.events and (now - self.events[0][0]).total_seconds() > (BASE_MIN + 1) * 60 + WINDOW_SEC:
            self.events.popleft()

        start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        elapsed = (now - start).total_seconds() / 60
        if elapsed < SKIP_OPEN_MIN or not self.price:
            return alerts

        win = [e for e in self.events if (now - e[0]).total_seconds() <= WINDOW_SEC]
        base = [e for e in self.events if WINDOW_SEC < (now - e[0]).total_seconds()]
        vol_win = sum(e[1] for e in win)
        base_min = max(sum(e[1] for e in base) / BASE_MIN, self.vma_lots / TRADING_MIN)
        if vol_win < MIN_LOTS or vol_win < SURGE_X * base_min:
            return alerts

        out = sum(e[2] for e in win)
        inn = sum(e[3] for e in win)
        ratio = out / (out + inn) if out + inn else 0.5
        dp = win[-1][4] - win[0][4]
        if ratio >= BUY_RATIO and dp >= 0:
            side = "buy"
        elif ratio <= SELL_RATIO and dp <= 0:
            side = "sell"
        else:
            side = "unclear"

        last = self.last_alert.get(side)
        if last and (now - last).total_seconds() < COOLDOWN_MIN * 60:
            return alerts
        self.last_alert[side] = now

        p = self.price
        mid, up, dn = self.bands(p)
        proj = v / max(elapsed, 1) * TRADING_MIN / self.vma_lots   # 預估全日量比
        chg = (p - self.prev_close) / self.prev_close * 100
        shadow = (self.high - p) / p * 100 if self.high else 0

        advice = []
        if side == "buy":
            if p > up and proj >= 1.3 and mid > self.prev_mid:
                advice.append("符合 A級突破：站上上軌＋放量，可考慮買進")
            if p > mid and abs(p - mid) / mid <= 0.01:
                advice.append("B級：中軌附近買盤湧入，回踩買點")
            if p > up and proj >= 2:
                advice.append("⚠️ 預估爆量 2 倍以上，別追高，留意 D級長上影")
            if not advice:
                advice.append("買盤湧入，但位置未達進場條件，先觀察")
            title = "🟢 即時爆量・偏買"
        elif side == "sell":
            if p < mid:
                advice.append("C級：中軌下方爆量賣壓，考慮減碼／出場")
            if self.high and self.high >= up * 0.99 and shadow >= 2:
                advice.append("D級：衝高接近上軌後爆量回落，可先賣 1/3~1/2")
            if self.prev_close > self.prev_up and p < up:
                advice.append("E級：昨日突破上軌，今天跌回上軌下方，疑似假突破")
            if not advice:
                advice.append("賣壓湧出，持股者提高警覺")
            title = "🔴 即時爆量・偏賣"
        else:
            advice.append("買賣力道接近，方向不明，等收盤確認")
            title = "🟡 即時爆量・方向不明"

        msg = (f"1分鐘量 {vol_win:,.0f} 張（平常 {vol_win / base_min:.1f} 倍）\n"
               f"外盤比 {ratio * 100:.0f}%｜現價 {p:.2f}（{chg:+.2f}%）\n"
               f"預估全日量比 {proj:.2f}\n"
               f"上軌 {up:.2f}｜中軌 {mid:.2f}｜下軌 {dn:.2f}\n"
               + "\n".join("→ " + a for a in advice)
               + f"\n{now:%H:%M:%S}")
        alerts.append((side, f"{self.name}({self.code}) {title}", msg))
        return alerts


def main():
    today = dt.datetime.now(TZ).date()
    end_h, end_m = map(int, os.getenv("END", "13:30").split(":"))
    stocks = parse_stocks()
    keys = resolve_keys([c for c, _ in stocks])

    dets = {}
    for code, name in stocks:
        if code not in keys:
            print(f"{code} 即時行情查不到，略過")
            continue
        daily = fetch(code)
        if daily is None:
            print(f"{code} 日K抓不到，略過")
            continue
        dets[code] = Detector(code, name, add_indicators(daily), today)
    if not dets:
        return
    key_list = [keys[c] for c in dets]
    print("開始監控：", ", ".join(f"{d.name}({c})" for c, d in dets.items()))

    errors = 0
    while True:
        now = dt.datetime.now(TZ)
        if (now.hour, now.minute) >= (end_h, end_m):
            print("收盤，結束監控")
            break
        if (now.hour, now.minute) < (9, 0):
            time.sleep(30)
            continue
        try:
            for m in mis_query(key_list):
                if m.get("d") != today.strftime("%Y%m%d"):
                    continue                         # 休市日資料不是今天
                det = dets.get(m["c"])
                if det:
                    for side, title, msg in det.update(m, now):
                        notify(title, msg, priority=5 if side == "sell" else 4,
                               tags=["rotating_light"])
            errors = 0
        except Exception as e:                       # 網路偶爾失敗就重試
            errors += 1
            print("抓取失敗：", e)
            time.sleep(min(60, 5 * errors))
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
