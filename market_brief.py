# -*- coding: utf-8 -*-
"""
글로벌 증시 데일리 브리핑 v2 (GitHub Actions + 텔레그램)

v1 대비 추가:
  - 이상 신호 탐지 (섹터 상관 붕괴, 팩터 스프레드, 거래대금 이상,
    리스크지표 역행, 지역 디커플링, 채권-주식 상관 반전)
  - 블로그 코멘트 소재를 별도 메시지로 전송

필요 Secrets: ANTHROPIC_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

MODEL = "claude-sonnet-4-6"
KST = dt.timezone(dt.timedelta(hours=9))

# 이상 탐지 파라미터
LOOKBACK = 60          # 통계 기준 기간 (거래일)
Z_THRESHOLD = 2.0      # 이 값을 넘으면 신호로 판정 (중간 민감도)
MAX_SIGNALS = 8        # 최대 표시 개수


def check_config():
    missing = [k for k in ("ANTHROPIC_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")
               if not os.environ.get(k)]
    if missing:
        print(f"[error] Secrets 미설정: {', '.join(missing)}")
        sys.exit(1)


# ═════════════════════════════════════════════
# 수집 대상
# ═════════════════════════════════════════════
INDICES = {
    "S&P 500": "^GSPC", "나스닥": "^IXIC", "다우": "^DJI",
    "러셀2000": "^RUT", "필라델피아 반도체": "^SOX", "VIX": "^VIX",
    "유로 STOXX 50": "^STOXX50E", "독일 DAX": "^GDAXI",
    "영국 FTSE100": "^FTSE", "프랑스 CAC40": "^FCHI", "일본 닛케이225": "^N225",
}

SECTORS = {
    "기술": "XLK", "금융": "XLF", "헬스케어": "XLV", "에너지": "XLE",
    "산업재": "XLI", "경기소비재": "XLY", "필수소비재": "XLP",
    "유틸리티": "XLU", "소재": "XLB", "부동산": "XLRE", "커뮤니케이션": "XLC",
}

# M7 (한국 투자자 최대 관심군)
M7 = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Alphabet": "GOOGL",
    "Amazon": "AMZN", "NVIDIA": "NVDA", "Meta": "META", "Tesla": "TSLA",
}

# 한국 투자자 관심 종목 (서학개미 순매수 상위권 위주)
KR_FAVORITES = {
    "Broadcom": "AVGO", "AMD": "AMD", "Micron": "MU",
    "TSMC(ADR)": "TSM", "ASML": "ASML", "Palantir": "PLTR",
    "Coinbase": "COIN", "MicroStrategy": "MSTR", "Rivian": "RIVN",
    "Eli Lilly": "LLY", "Novo Nordisk": "NVO", "Netflix": "NFLX",
    "ARM": "ARM", "Super Micro": "SMCI", "Vertiv": "VRT",
    "Oracle": "ORCL", "Salesforce": "CRM", "Uber": "UBER",
}

# 밈·고변동성 종목 (한국 개인 거래 활발)
MEME = {
    "GameStop": "GME", "AMC": "AMC", "Robinhood": "HOOD",
    "SoFi": "SOFI", "Lucid": "LCID", "Plug Power": "PLUG",
    "IonQ": "IONQ", "Rigetti": "RGTI",
}

# 레버리지 ETF (한국 개인 대량 보유)
LEVERAGED = {
    "TQQQ(나스닥3배)": "TQQQ", "SOXL(반도체3배)": "SOXL",
    "TSLL(테슬라2배)": "TSLL", "NVDL(엔비디아2배)": "NVDL",
}

# 거래대금 이상 탐지 대상 (M7 + 관심종목 + 밈)
WATCHLIST = {**M7, **KR_FAVORITES, **MEME}

# 미국 매크로 지표 (경기·금리 판단용)
US_MACRO = {
    "미 10년물 금리": "^TNX",
    "미 2년물 금리": "^IRX",
    "미 30년물 금리": "^TYX",
    "달러인덱스": "DX-Y.NYB",
    "VIX(변동성)": "^VIX",
    "미국채 장기(TLT)": "TLT",
    "물가연동채(TIP)": "TIP",
    "하이일드(HYG)": "HYG",
    "투자등급회사채(LQD)": "LQD",
    "지역은행(KRE)": "KRE",
    "주택건설(XHB)": "XHB",
    "운송(IYT)": "IYT",
    "소매(XRT)": "XRT",
}

MACRO = {
    "미 10년물 금리": "^TNX", "미 2년물 금리": "^IRX", "달러인덱스": "DX-Y.NYB",
    "원/달러": "KRW=X", "달러/엔": "JPY=X", "유로/달러": "EURUSD=X",
    "WTI 유가": "CL=F", "브렌트유": "BZ=F", "금": "GC=F",
    "구리": "HG=F", "비트코인": "BTC-USD",
}

KOSPI = {"코스피": "^KS11", "코스닥": "^KQ11"}

# 팩터 스프레드 관찰 쌍 (이름, 티커A, 티커B, 설명)
FACTOR_PAIRS = [
    ("성장 vs 가치", "IVW", "IVE", "성장주와 가치주"),
    ("대형 vs 소형", "SPY", "IWM", "대형주와 소형주"),
    ("모멘텀 vs 로우볼", "MTUM", "USMV", "모멘텀과 저변동성"),
    ("퀄리티 vs 시장", "QUAL", "SPY", "퀄리티 팩터와 시장"),
]

# 지역 디커플링 관찰 쌍
REGION_PAIRS = [
    ("미국 vs 신흥국", "SPY", "EEM", "미국과 신흥국"),
    ("한국 vs 대만", "EWY", "EWT", "한국과 대만"),
    ("미국 vs 유럽", "SPY", "VGK", "미국과 유럽"),
    ("한국 vs 미국", "EWY", "SPY", "한국과 미국"),
]

# 리스크 지표 (주식과 방향이 갈리면 신호)
RISK_ASSETS = {
    "미국채 장기(TLT)": "TLT",
    "금(GLD)": "GLD",
    "달러(UUP)": "UUP",
    "하이일드(HYG)": "HYG",
}

NEWS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.investing.com/rss/news_25.rss",
]


# ═════════════════════════════════════════════
# 데이터 수집
# ═════════════════════════════════════════════
def download(tickers, period="6mo"):
    """티커 리스트의 OHLCV 다운로드"""
    try:
        return yf.download(
            " ".join(tickers), period=period, interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        print(f"[warn] 다운로드 실패: {e}")
        return None


def extract_close(data, ticker, multi=True):
    """종가 시리즈 추출"""
    try:
        if multi:
            return data[ticker]["Close"].dropna()
        return data["Close"].dropna()
    except Exception:
        return None


def fetch_quotes(mapping, label=""):
    rows = []
    tickers = list(mapping.values())
    data = download(tickers, period="3mo")
    if data is None:
        return rows

    for name, tkr in mapping.items():
        close = extract_close(data, tkr, multi=len(tickers) > 1)
        if close is None or len(close) < 2:
            continue
        last, prev = float(close.iloc[-1]), float(close.iloc[-2])
        rows.append({
            "name": name, "ticker": tkr, "last": last,
            "chg_pct": (last / prev - 1) * 100,
            "date": close.index[-1].strftime("%Y-%m-%d(%a)"),
        })
    return rows


def fmt_rows(title, rows):
    if not rows:
        return f"[{title}]\n(데이터 없음)\n"
    out = [f"[{title}]"]
    for r in rows:
        out.append(f"- {r['name']}: {r['last']:,.2f} ({r['chg_pct']:+.2f}%) [{r['date']}]")
    return "\n".join(out) + "\n"


# ═════════════════════════════════════════════
# 이상 신호 탐지
# ═════════════════════════════════════════════
def zscore(series_hist, value_today):
    """과거 분포 대비 오늘 값의 z-score"""
    s = pd.Series(series_hist).dropna()
    if len(s) < 20:
        return None
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return None
    return float((value_today - s.mean()) / sd)


def detect_pair_divergence(returns_df, pairs, category):
    """두 자산 간 스프레드가 평소 범위를 벗어난 경우 탐지"""
    signals = []
    for label, a, b, desc in pairs:
        if a not in returns_df.columns or b not in returns_df.columns:
            continue
        sub = returns_df[[a, b]].dropna()
        if len(sub) < LOOKBACK // 2:
            continue

        hist = sub.iloc[:-1].tail(LOOKBACK)
        today = sub.iloc[-1]

        spread_hist = hist[a] - hist[b]
        spread_today = float(today[a] - today[b])
        z = zscore(spread_hist, spread_today)
        if z is None or abs(z) < Z_THRESHOLD:
            continue

        corr = float(hist[a].corr(hist[b]))
        signals.append({
            "category": category,
            "label": label,
            "desc": desc,
            "z": z,
            "detail": (
                f"{desc} 격차 {spread_today*100:+.2f}%p "
                f"(평소 {LOOKBACK}일 기준 z={z:+.1f}, 상관계수 {corr:.2f})"
            ),
            "a_ret": float(today[a]) * 100,
            "b_ret": float(today[b]) * 100,
            "a": a, "b": b,
        })
    return signals


def detect_sector_decoupling(returns_df, sector_map):
    """섹터 간 상관이 평소와 크게 달라진 경우 — 가장 벌어진 쌍"""
    cols = [t for t in sector_map.values() if t in returns_df.columns]
    if len(cols) < 4:
        return []

    sub = returns_df[cols].dropna()
    if len(sub) < LOOKBACK // 2:
        return []

    hist = sub.iloc[:-1].tail(LOOKBACK)
    today = sub.iloc[-1]
    rev = {v: k for k, v in sector_map.items()}

    best = None
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            corr = float(hist[a].corr(hist[b]))
            if corr < 0.5:          # 평소 같이 움직이던 쌍만 대상
                continue
            spread_hist = hist[a] - hist[b]
            spread_today = float(today[a] - today[b])
            z = zscore(spread_hist, spread_today)
            if z is None or abs(z) < Z_THRESHOLD:
                continue
            if best is None or abs(z) > abs(best["z"]):
                best = {
                    "category": "섹터 상관 붕괴",
                    "label": f"{rev[a]} vs {rev[b]}",
                    "desc": f"{rev[a]}와 {rev[b]}",
                    "z": z,
                    "detail": (
                        f"평소 상관 {corr:.2f}로 동조하던 {rev[a]}({today[a]*100:+.2f}%)와 "
                        f"{rev[b]}({today[b]*100:+.2f}%)가 {spread_today*100:+.2f}%p 벌어짐 (z={z:+.1f})"
                    ),
                    "a_ret": float(today[a]) * 100,
                    "b_ret": float(today[b]) * 100,
                    "a": a, "b": b,
                }
    return [best] if best else []


def detect_stock_moves(data, mapping, category="개별 종목 급등락"):
    """개별 종목의 이례적인 일간 변동 탐지"""
    signals = []
    tickers = list(mapping.values())
    rev = {v: k for k, v in mapping.items()}

    for tkr in tickers:
        try:
            df = data[tkr] if len(tickers) > 1 else data
            close = df["Close"].dropna()
            if len(close) < LOOKBACK:
                continue

            rets = close.pct_change().dropna()
            hist = rets.iloc[:-1].tail(LOOKBACK)
            today_ret = float(rets.iloc[-1])
            z = zscore(hist, today_ret)
            if z is None or abs(z) < Z_THRESHOLD:
                continue

            signals.append({
                "category": category,
                "label": rev[tkr],
                "desc": rev[tkr],
                "z": z,
                "detail": (
                    f"{rev[tkr]} {today_ret*100:+.2f}% "
                    f"(최근 {LOOKBACK}일 변동성 대비 z={z:+.1f}, "
                    f"평소 일간 변동폭 ±{hist.std()*100:.1f}%)"
                ),
                "ret": today_ret * 100,
            })
        except Exception:
            continue

    signals.sort(key=lambda s: abs(s["z"]), reverse=True)
    return signals[:4]


def detect_leverage_gap(data_lev, data_m7):
    """레버리지 ETF와 기초자산의 괴리 (한국 개인 보유 많음)"""
    signals = []
    pairs = [
        ("TSLL(테슬라2배)", "TSLL", "TSLA", 2.0),
        ("NVDL(엔비디아2배)", "NVDL", "NVDA", 2.0),
    ]
    for label, lev_t, base_t, mult in pairs:
        try:
            lc = data_lev[lev_t]["Close"].dropna() if lev_t in data_lev else None
            bc = data_m7[base_t]["Close"].dropna() if base_t in data_m7 else None
            if lc is None or bc is None or len(lc) < 5 or len(bc) < 5:
                continue
            lev_ret = (float(lc.iloc[-1]) / float(lc.iloc[-2]) - 1) * 100
            base_ret = (float(bc.iloc[-1]) / float(bc.iloc[-2]) - 1) * 100
            expected = base_ret * mult
            gap = lev_ret - expected
            if abs(gap) < 0.8 or abs(base_ret) < 1.0:
                continue
            signals.append({
                "category": "레버리지 ETF 괴리",
                "label": label,
                "desc": label,
                "z": abs(gap),
                "detail": (
                    f"{label} {lev_ret:+.2f}%, 기초자산 {base_ret:+.2f}% "
                    f"(이론상 {expected:+.2f}%, 괴리 {gap:+.2f}%p)"
                ),
            })
        except Exception:
            continue
    return signals


def detect_volume_anomaly(data, mapping):
    """가격 변동 대비 거래대금이 이례적으로 튄 종목"""
    signals = []
    tickers = list(mapping.values())
    rev = {v: k for k, v in mapping.items()}

    for tkr in tickers:
        try:
            df = data[tkr] if len(tickers) > 1 else data
            close = df["Close"].dropna()
            vol = df["Volume"].dropna()
            if len(close) < LOOKBACK or len(vol) < LOOKBACK:
                continue

            dv = (close * vol).dropna()
            hist = dv.iloc[:-1].tail(LOOKBACK)
            today_dv = float(dv.iloc[-1])
            z = zscore(np.log(hist[hist > 0]), np.log(today_dv) if today_dv > 0 else 0)
            if z is None or z < Z_THRESHOLD:
                continue

            ret = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
            ratio = today_dv / float(hist.mean()) if hist.mean() > 0 else 0

            signals.append({
                "category": "거래대금 이상",
                "label": rev[tkr],
                "desc": rev[tkr],
                "z": z,
                "detail": (
                    f"{rev[tkr]} 거래대금이 평소 대비 {ratio:.1f}배 (z={z:+.1f}), "
                    f"같은 날 주가는 {ret:+.2f}%"
                ),
                "ret": ret,
                "ratio": ratio,
            })
        except Exception:
            continue

    signals.sort(key=lambda s: s["z"], reverse=True)
    return signals[:3]


def detect_risk_divergence(returns_df, equity_ticker="SPY"):
    """주식과 리스크 지표가 평소와 다른 방향으로 간 경우"""
    signals = []
    if equity_ticker not in returns_df.columns:
        return signals

    for name, tkr in RISK_ASSETS.items():
        if tkr not in returns_df.columns:
            continue
        sub = returns_df[[equity_ticker, tkr]].dropna()
        if len(sub) < LOOKBACK // 2:
            continue

        hist = sub.iloc[:-1].tail(LOOKBACK)
        today = sub.iloc[-1]
        corr = float(hist[equity_ticker].corr(hist[tkr]))

        eq_ret = float(today[equity_ticker]) * 100
        rk_ret = float(today[tkr]) * 100

        # 평소 상관의 부호와 오늘 동시 움직임의 부호가 반대인 경우
        today_sign = np.sign(eq_ret * rk_ret)
        if abs(corr) < 0.25 or today_sign == np.sign(corr) or abs(eq_ret) < 0.3:
            continue

        spread_hist = hist[equity_ticker] - hist[tkr]
        z = zscore(spread_hist, float(today[equity_ticker] - today[tkr]))
        if z is None or abs(z) < Z_THRESHOLD:
            continue

        signals.append({
            "category": "리스크 지표 역행",
            "label": f"주식 vs {name}",
            "desc": f"주식과 {name}",
            "z": z,
            "detail": (
                f"평소 상관 {corr:+.2f}인 S&P500({eq_ret:+.2f}%)과 {name}({rk_ret:+.2f}%)이 "
                f"반대 방향 (z={z:+.1f})"
            ),
        })
    return signals


def build_returns_frame():
    """이상 탐지에 필요한 전 티커의 일간 수익률 프레임 구성"""
    tickers = set()
    tickers.update(SECTORS.values())
    for _, a, b, _ in FACTOR_PAIRS + REGION_PAIRS:
        tickers.update([a, b])
    tickers.update(RISK_ASSETS.values())
    tickers.add("SPY")
    tickers = sorted(tickers)

    data = download(tickers, period="9mo")
    if data is None:
        return None, None

    closes = {}
    for t in tickers:
        c = extract_close(data, t, multi=len(tickers) > 1)
        if c is not None and len(c) > LOOKBACK:
            closes[t] = c

    if not closes:
        return None, None

    px = pd.DataFrame(closes)
    returns = px.pct_change().dropna(how="all")
    return returns, data


def detect_all_signals():
    """모든 이상 신호 탐지 후 강도순 정렬"""
    print("  이상 신호 탐지 중...")
    returns, _ = build_returns_frame()

    signals = []
    if returns is not None:
        signals += detect_sector_decoupling(returns, SECTORS)
        signals += detect_pair_divergence(returns, FACTOR_PAIRS, "팩터 스프레드")
        signals += detect_pair_divergence(returns, REGION_PAIRS, "지역 디커플링")
        signals += detect_risk_divergence(returns)

    # 개별 종목: 급등락 + 거래대금 이상
    stock_data = download(list(WATCHLIST.values()), period="6mo")
    if stock_data is not None:
        signals += detect_stock_moves(stock_data, M7, "M7 급등락")
        signals += detect_stock_moves(stock_data, KR_FAVORITES, "관심종목 급등락")
        signals += detect_stock_moves(stock_data, MEME, "밈주식 급등락")
        signals += detect_volume_anomaly(stock_data, WATCHLIST)

    # 레버리지 ETF 괴리
    lev_data = download(list(LEVERAGED.values()), period="3mo")
    if lev_data is not None and stock_data is not None:
        signals += detect_leverage_gap(lev_data, stock_data)

    # 미국 매크로 지표 급변
    macro_data = download(list(US_MACRO.values()), period="6mo")
    if macro_data is not None:
        signals += detect_stock_moves(macro_data, US_MACRO, "미국 매크로 급변")

    # 중복 제거 (같은 종목이 급등락과 거래대금 양쪽에 걸린 경우 강한 쪽만)
    seen, dedup = {}, []
    for s in sorted(signals, key=lambda x: abs(x["z"]), reverse=True):
        key = s["label"]
        if key in seen:
            continue
        seen[key] = True
        dedup.append(s)

    return dedup[:MAX_SIGNALS]


def fmt_signals(signals):
    if not signals:
        return "[이상 신호]\n오늘은 통계적으로 특이한 움직임이 감지되지 않았습니다.\n"
    lines = ["[이상 신호 — 통계 기준 이탈 항목]"]
    for i, s in enumerate(signals, 1):
        lines.append(f"{i}. ({s['category']}) {s['detail']}")
    return "\n".join(lines) + "\n"


# ═════════════════════════════════════════════
# 뉴스
# ═════════════════════════════════════════════
def fetch_news(limit_per_feed=8):
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=30)
    items = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"[warn] RSS 실패 {url}: {e}")
            continue
        for entry in feed.entries[:limit_per_feed]:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub:
                pdt = dt.datetime(*pub[:6], tzinfo=dt.timezone.utc)
                if pdt < cutoff:
                    continue
                stamp = pdt.strftime("%m-%d %H:%M UTC")
            else:
                stamp = ""
            summary = (entry.get("summary") or "")[:300].replace("\n", " ")
            items.append(f"- ({stamp}) {title} | {summary}")
    return "\n".join(items[:60]) if items else "(뉴스 수집 실패)"


# ═════════════════════════════════════════════
# 원고 생성
# ═════════════════════════════════════════════
BRIEF_PROMPT = """당신은 한국 투자자를 위한 데일리 글로벌 증시 브리핑을 쓰는 애널리스트입니다.
독자는 미국 주식과 ETF를 직접 사고 있는 한국 개인투자자입니다. 이들의 관심사는 명확합니다.
M7과 반도체, 본인이 들고 있는 개별 종목, 그리고 밤사이 미국 매크로가 오늘 한국장에 어떤 영향을 줄지입니다.

작성 원칙:
- 한국어로 작성. 문어체가 아닌 평이하고 직설적인 문장. 미사여구 금지.
- 숫자는 반드시 제공된 데이터에서만 인용. 없는 숫자를 지어내지 말 것.
- 데이터에 없는 사실은 쓰지 말 것. 특히 날짜와 요일을 임의로 추론하지 말 것.
- 해석을 덧붙이되 단정하지 말 것. "~로 보인다", "~가 배경으로 지목된다" 수준.
- 투자 권유, 목표주가, 매수/매도 의견 금지.
- 네이버 블로그에 그대로 붙여넣을 형태로 출력. 마크다운 문법(##, **) 대신 일반 텍스트와 줄바꿈 사용.

분량 배분이 중요합니다. 전체의 절반 이상을 개별 종목과 미국 매크로에 쓰세요.
지수 마감 숫자 나열에 분량을 쓰지 마세요. 그건 어디에나 있는 정보입니다.

출력 구조:

1) 제목 한 줄

2) 세 줄 요약 — 오늘 한국 투자자가 반드시 알아야 할 것

3) M7과 주요 관심 종목 (가장 비중 크게)
   - M7 중 움직임이 컸던 종목을 중심으로. 전 종목을 나열하지 말 것.
   - 반도체(NVDA, AMD, MU, TSM, ASML, AVGO), AI 인프라(PLTR, SMCI, VRT, ARM),
     비만치료제(LLY, NVO) 등 한국 투자자 보유 비중이 높은 종목을 우선 다룰 것.
   - 급등락한 종목이 있으면 그 종목에 문단을 할애할 것.
   - 밈·고변동성 종목(GME, HOOD, IONQ, RGTI 등)에 큰 움직임이 있으면 별도로 언급.
   - 레버리지 ETF(TQQQ, SOXL, TSLL, NVDL)는 한국 개인 보유가 많으므로
     기초자산 대비 움직임을 언급할 가치가 있음.

4) 미국 매크로 (두 번째로 비중 크게)
   - 금리 곡선(2년/10년/30년) 움직임과 그 의미
   - 달러인덱스, VIX
   - 크레딧 신호: 하이일드(HYG)와 투자등급(LQD)의 방향
   - 경기 민감 섹터 신호: 지역은행(KRE), 주택건설(XHB), 운송(IYT), 소매(XRT)
     이 지표들이 무엇을 시사하는지 짚을 것
   - 매크로 지표 간 방향이 엇갈리면 그 점을 명시할 것

5) 지수·섹터 마감 (짧게)
   - 미국·유럽 주요 지수는 간결하게. 숫자 나열 최소화.
   - 섹터는 상위/하위 몇 개만.

6) 환율·원자재

7) 오늘 한국시장 관전포인트
   - 위 내용이 한국장에 어떻게 연결될지
   - 특히 반도체·2차전지 등 한국 주력 업종과 연결되는 부분

8) 맨 아래 한 줄: "※ 본 글은 공개 데이터를 정리한 것으로 투자 권유가 아닙니다."

분량은 전체 1,800~2,500자."""


SIGNAL_PROMPT = """당신은 시장 분석 전문가의 블로그 집필을 돕는 보조입니다.
필자는 글로벌 지수회사에서 10년간 리스크 모델, 지수 전략, ESG, 기후, 사모자산 솔루션을 다뤄온 전문가입니다.
독자는 미국 주식과 ETF를 직접 사고 있는 한국 개인투자자로, 수익률 추천에는 질렸고
시장이 왜 그렇게 움직이는지, 그게 앞으로 무엇을 의미하는지 알고 싶어하는 층입니다.

주어진 것은 통계적으로 평소 범위를 벗어난 항목들입니다.
아래 두 파트를 순서대로 작성하세요.

═══ PART 1. 판단 재료 ═══
각 신호마다 다음 세 가지를 간결하게 정리합니다. 전문가 본인이 읽고 판단하는 용도이므로 기초 설명은 생략합니다.

- 관찰: 무슨 일이 있었는지 (제공된 숫자만 사용)
- 가능한 해석: 이 움직임을 설명할 수 있는 방향 2~3가지. 병렬로 제시하고 단정하지 말 것.
- 확인할 점: 어느 해석이 맞는지 가리려면 무엇을 더 봐야 하는지

═══ PART 2. 블로그 코멘트 초안 ═══
PART 1에서 가장 이야깃거리가 되는 신호 하나(많아야 둘)를 골라, 블로그에 그대로 붙여넣을 수 있는 코멘트를 씁니다.
서로 다른 각도의 초안 2개를 제시하세요. (A안 / B안)

★ 각도 선택이 가장 중요합니다.
신호의 성격에 맞는 각도를 고르세요. 모든 신호를 지수나 팩터 이야기로 끌고 가지 마세요.
개별 종목의 급등락을 억지로 지수·ETF·리밸런싱 프레임에 연결하는 것은 하지 마세요.
그런 연결은 실제로 그 요인이 작동했다는 근거가 데이터에 있을 때만 하세요.

각도를 고르는 기준:
- 개별 종목의 큰 움직임이면 → 그 사안 자체가 앞으로 미국 시장과 한국 시장에
  어떤 영향을 줄 가능성이 있는지를 쓸 것. 해당 종목의 밸류체인, 경쟁 구도,
  같은 업종 한국 기업에 미칠 파급을 보는 것이 자연스럽습니다.
- 매크로 지표 간 엇갈림이면 → 경기·금리 국면에 대한 함의
- 여러 종목·섹터에 걸친 광범위한 움직임이면 → 이때만 팩터·지수 관점이 적절
- 거래대금 이상이면 → 수급 관점
- 지역 간 디커플링이면 → 한국 시장에 어떻게 연결되는지

A안과 B안은 서로 다른 각도여야 합니다. 같은 이야기를 두 번 쓰지 마세요.

각 초안의 작성 규칙:
- 분량은 4~6문장.
- 문체는 조심스럽게. "~로 보입니다", "~일 가능성이 있습니다", "~는 아직 단정하기 어렵습니다".
  단정형으로 인과를 확정하지 말 것.
- 전문용어는 그대로 쓰되 처음 나올 때 괄호로 짧게 풀이할 것. 한 코멘트에 1~2개까지만.
- 첫 문장은 관찰된 사실로 시작. 배경 설명부터 시작하지 말 것.
- 중간에 "그래서 이게 무슨 의미인가"를 반드시 담을 것. 현상 서술로 끝나면 안 됩니다.
- 마지막 문장은 독자가 앞으로 무엇을 볼지 알려주는 문장으로.
- 한국 투자자가 읽는 글이므로, 가능하면 한국 시장·한국 기업과의 연결점을 언급할 것.
  단, 억지로 만들지는 말 것. 연결이 자연스럽지 않으면 미국 시장 이야기로만 끝내도 됩니다.
- 특정 종목이나 ETF의 매수·매도를 권하는 문장은 절대 쓰지 말 것.
- 목표가, 미래 수익률 전망 금지.

═══ 공통 원칙 ═══
- 한국어.
- 제공된 데이터에 없는 숫자, 사건, 뉴스를 절대 만들어내지 말 것.
- 뉴스 헤드라인이 주어진 경우 시점이 맞는 것만 조심스럽게 연결. 억지로 엮지 말 것.
- 신호가 없으면 PART 2는 생략하고 그렇다고 쓸 것."""


def call_claude(system, user, max_tokens=3000):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# ═════════════════════════════════════════════
# 텔레그램
# ═════════════════════════════════════════════
TG_LIMIT = 4000


def split_text(text, limit=TG_LIMIT):
    chunks, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > limit:
            chunks.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i, chunk in enumerate(split_text(text), 1):
        r = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID, "text": chunk,
            "disable_web_page_preview": True,
        }, timeout=30)
        if not r.ok:
            print(f"[error] 전송 실패({i}): {r.status_code} {r.text}")
            r.raise_for_status()
        time.sleep(0.4)


def send_document(filename, content, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        r = requests.post(
            url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
            files={"document": (filename, content.encode("utf-8"), "text/plain")},
            timeout=60,
        )
        if not r.ok:
            print(f"[warn] 첨부 실패: {r.text}")
    except Exception as e:
        print(f"[warn] 첨부 실패: {e}")


# ═════════════════════════════════════════════
# 메인
# ═════════════════════════════════════════════
def main():
    check_config()

    now = dt.datetime.now(KST)
    today = now.strftime("%Y-%m-%d(%a)")
    print(f"=== 데일리 브리핑 v2 시작: {today} KST ===")

    print("1/5 시장 데이터 수집...")
    idx = fetch_quotes(INDICES, "지수")
    sec = fetch_quotes(SECTORS, "섹터")
    m7 = fetch_quotes(M7, "M7")
    fav = fetch_quotes(KR_FAVORITES, "관심종목")
    meme = fetch_quotes(MEME, "밈주식")
    lev = fetch_quotes(LEVERAGED, "레버리지ETF")
    usm = fetch_quotes(US_MACRO, "미국매크로")
    mac = fetch_quotes(MACRO, "환율원자재")
    kor = fetch_quotes(KOSPI, "한국")

    data_block = "\n".join([
        fmt_rows("글로벌 주요 지수", idx),
        fmt_rows("M7 (한국 투자자 최대 관심군)", m7),
        fmt_rows("한국 투자자 관심 종목", fav),
        fmt_rows("밈·고변동성 종목", meme),
        fmt_rows("레버리지 ETF (한국 개인 대량 보유)", lev),
        fmt_rows("미국 매크로 지표", usm),
        fmt_rows("미국 섹터 ETF", sec),
        fmt_rows("환율·원자재", mac),
        fmt_rows("한국 지수(전 거래일)", kor),
    ])

    print("2/5 이상 신호 탐지...")
    signals = detect_all_signals()
    signal_block = fmt_signals(signals)
    print(f"  → {len(signals)}개 감지")

    print("3/5 뉴스 수집...")
    news_block = fetch_news()

    print("4/5 원고 생성...")
    article = call_claude(
        BRIEF_PROMPT,
        f"아래는 {today} 기준 수집 데이터입니다. 블로그 원고를 작성해 주세요.\n\n"
        f"=== 시장 데이터 ===\n{data_block}\n\n"
        f"=== 주요 뉴스 ===\n{news_block}\n\n"
        f"위 데이터만 사용하세요. 없는 수치나 사건을 만들지 마세요.",
        max_tokens=3000,
    )

    if signals:
        material = call_claude(
            SIGNAL_PROMPT,
            f"{today} 기준 감지된 이상 신호입니다.\n\n"
            f"{signal_block}\n"
            f"=== 참고: 같은 날 시장 데이터 ===\n{data_block}\n\n"
            f"=== 참고: 최근 뉴스 헤드라인 ===\n{news_block[:3000]}\n\n"
            f"PART 1(판단 재료)과 PART 2(블로그 코멘트 초안 A안/B안)를 작성해 주세요.",
            max_tokens=3500,
        )
    else:
        material = (
            "오늘은 통계적으로 특이한 움직임이 감지되지 않았습니다.\n"
            "평소 범위 안에서 움직인 날도 그 자체로 관찰 대상입니다."
        )

    print("5/5 전송...")
    send_telegram(f"[증시 브리핑] {today}\n\n{article}")
    time.sleep(1)
    send_telegram(f"━━━━━━━━━━━━━━━\n[코멘트 소재 + 초안] {today}\n━━━━━━━━━━━━━━━\n\n{material}")

    send_document(
        f"brief_{now.strftime('%Y%m%d')}.txt",
        (article + "\n\n" + "=" * 60 + "\n[코멘트 소재]\n" + "=" * 60 + "\n" + material
         + "\n\n" + "=" * 60 + "\n[탐지된 신호 원본]\n" + "=" * 60 + "\n" + signal_block
         + "\n\n" + "=" * 60 + "\n[수집 원본 데이터]\n" + "=" * 60 + "\n" + data_block),
        caption="원고 + 코멘트 소재 + 원본 데이터",
    )

    print("완료.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            send_telegram(f"[브리핑 실패] {dt.datetime.now(KST):%Y-%m-%d}\n\n{traceback.format_exc()[-3000:]}")
        except Exception:
            pass
        raise
