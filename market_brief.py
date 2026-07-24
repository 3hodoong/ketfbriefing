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
MAX_SIGNALS = 6        # 최대 표시 개수


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

WATCHLIST = {
    "NVIDIA": "NVDA", "Micron": "MU", "TSMC(ADR)": "TSM", "ASML": "ASML",
    "Apple": "AAPL", "Microsoft": "MSFT", "Tesla": "TSLA", "Broadcom": "AVGO",
    "AMD": "AMD", "Meta": "META", "Alphabet": "GOOGL", "Amazon": "AMZN",
    "Eli Lilly": "LLY", "Exxon Mobil": "XOM", "JPMorgan": "JPM",
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
    if returns is None:
        return []

    signals = []
    signals += detect_sector_decoupling(returns, SECTORS)
    signals += detect_pair_divergence(returns, FACTOR_PAIRS, "팩터 스프레드")
    signals += detect_pair_divergence(returns, REGION_PAIRS, "지역 디커플링")
    signals += detect_risk_divergence(returns)

    # 거래대금은 개별종목 데이터가 따로 필요
    vol_data = download(list(WATCHLIST.values()), period="6mo")
    if vol_data is not None:
        signals += detect_volume_anomaly(vol_data, WATCHLIST)

    signals.sort(key=lambda s: abs(s["z"]), reverse=True)
    return signals[:MAX_SIGNALS]


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
BRIEF_PROMPT = """당신은 한국 금융시장 실무자를 위한 데일리 글로벌 증시 브리핑을 쓰는 애널리스트입니다.

작성 원칙:
- 한국어로 작성. 문어체가 아닌 평이하고 직설적인 문장. 미사여구 금지.
- 숫자는 반드시 제공된 데이터에서만 인용. 없는 숫자를 지어내지 말 것.
- 데이터에 없는 사실은 쓰지 말 것. 특히 날짜와 요일을 임의로 추론하지 말 것.
- 해석을 덧붙이되 단정하지 말 것. "~로 보인다", "~가 배경으로 지목된다" 수준.
- 투자 권유, 목표주가, 매수/매도 의견 금지.
- 네이버 블로그에 그대로 붙여넣을 형태로 출력. 마크다운 문법(##, **) 대신 일반 텍스트와 줄바꿈 사용.

출력 구조:
1) 제목 한 줄
2) 세 줄 요약
3) 미국·유럽 지수 마감
4) 섹터별 등락과 특징주
5) 금리·환율·유가·금
6) 당일 한국시장 관전포인트
7) 맨 아래 한 줄: "※ 본 글은 공개 데이터를 정리한 것으로 투자 권유가 아닙니다."

분량은 전체 1,200~1,800자."""


SIGNAL_PROMPT = """당신은 지수·리스크모델 전문가를 보조하는 애널리스트입니다.
사용자는 MSCI에서 리스크 모델, 지수 전략, ESG, 기후, 사모자산 솔루션을 다뤄온 전문가이며,
본인 블로그에 직접 쓸 짧은 코멘트의 '소재'를 찾고 있습니다.

주어진 것은 통계적으로 평소 범위를 벗어난 항목들입니다.
당신의 역할은 코멘트를 대신 써주는 것이 아니라, 판단에 필요한 재료를 정리하는 것입니다.

각 신호마다 다음 세 가지를 제시하세요:
- 관찰: 무슨 일이 있었는지 (제공된 숫자만 사용)
- 가능한 해석: 이 움직임을 설명할 수 있는 방향을 2~3가지. 단정하지 말고 병렬로 제시.
  가능하면 지수·팩터·리밸런싱·수급 관점을 포함할 것.
- 확인할 점: 이 해석 중 어느 쪽인지 가리려면 무엇을 더 봐야 하는지

작성 원칙:
- 한국어. 간결하고 건조하게. 전문가 대상이므로 기초 설명 불필요.
- 제공된 데이터에 없는 숫자, 사건, 뉴스를 절대 만들어내지 말 것.
- 해석은 '가능성'으로만 제시. 확정적 인과 서술 금지.
- 뉴스 헤드라인이 주어진 경우, 신호와 시점이 맞는 것만 조심스럽게 연결. 억지로 엮지 말 것.
- 신호가 없으면 그렇다고 쓸 것.

분량은 신호당 4~6줄."""


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
    wat = fetch_quotes(WATCHLIST, "특징주")
    mac = fetch_quotes(MACRO, "매크로")
    kor = fetch_quotes(KOSPI, "한국")

    data_block = "\n".join([
        fmt_rows("글로벌 주요 지수", idx),
        fmt_rows("미국 섹터 ETF", sec),
        fmt_rows("주요 개별종목", wat),
        fmt_rows("금리·환율·원자재", mac),
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
            f"각 신호에 대해 관찰 / 가능한 해석 / 확인할 점을 정리해 주세요.",
            max_tokens=2500,
        )
    else:
        material = "오늘은 통계적으로 특이한 움직임이 감지되지 않았습니다.\n평범한 날도 그 자체로 관찰 대상입니다."

    print("5/5 전송...")
    send_telegram(f"[증시 브리핑] {today}\n\n{article}")
    time.sleep(1)
    send_telegram(f"━━━━━━━━━━━━━━━\n[코멘트 소재] {today}\n━━━━━━━━━━━━━━━\n\n{material}")

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
