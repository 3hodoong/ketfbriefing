# -*- coding: utf-8 -*-
"""
글로벌 증시 데일리 브리핑 자동 생성 스크립트 (GitHub Actions + 텔레그램 버전)

필요 Secrets: ANTHROPIC_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import sys
import datetime as dt

import requests
import yfinance as yf
import feedparser
import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

MODEL = "claude-sonnet-4-6"


def check_config():
    missing = [k for k in ("ANTHROPIC_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")
               if not os.environ.get(k)]
    if missing:
        print(f"[error] 다음 Secrets가 설정되지 않았습니다: {', '.join(missing)}")
        sys.exit(1)


# ─────────────────────────────────────────────
# 수집 대상
# ─────────────────────────────────────────────
INDICES = {
    "S&P 500": "^GSPC",
    "나스닥": "^IXIC",
    "다우": "^DJI",
    "러셀2000": "^RUT",
    "필라델피아 반도체": "^SOX",
    "VIX": "^VIX",
    "유로 STOXX 50": "^STOXX50E",
    "독일 DAX": "^GDAXI",
    "영국 FTSE100": "^FTSE",
    "프랑스 CAC40": "^FCHI",
    "일본 닛케이225": "^N225",
}

SECTORS = {
    "기술(XLK)": "XLK",
    "금융(XLF)": "XLF",
    "헬스케어(XLV)": "XLV",
    "에너지(XLE)": "XLE",
    "산업재(XLI)": "XLI",
    "경기소비재(XLY)": "XLY",
    "필수소비재(XLP)": "XLP",
    "유틸리티(XLU)": "XLU",
    "소재(XLB)": "XLB",
    "부동산(XLRE)": "XLRE",
    "커뮤니케이션(XLC)": "XLC",
}

WATCHLIST = {
    "NVIDIA": "NVDA",
    "Micron": "MU",
    "TSMC(ADR)": "TSM",
    "ASML": "ASML",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Broadcom": "AVGO",
    "AMD": "AMD",
    "Meta": "META",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Eli Lilly": "LLY",
    "Exxon Mobil": "XOM",
    "JPMorgan": "JPM",
}

MACRO = {
    "미 10년물 금리": "^TNX",
    "미 2년물 금리": "^IRX",
    "달러인덱스": "DX-Y.NYB",
    "원/달러": "KRW=X",
    "달러/엔": "JPY=X",
    "유로/달러": "EURUSD=X",
    "WTI 유가": "CL=F",
    "브렌트유": "BZ=F",
    "금": "GC=F",
    "구리": "HG=F",
    "비트코인": "BTC-USD",
}

KOSPI = {
    "코스피": "^KS11",
    "코스닥": "^KQ11",
}

NEWS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.investing.com/rss/news_25.rss",
]


# ─────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────
def fetch_quotes(mapping, label=""):
    rows = []
    tickers = " ".join(mapping.values())
    try:
        data = yf.download(
            tickers, period="7d", interval="1d",
            progress=False, auto_adjust=False, group_by="ticker",
        )
    except Exception as e:
        print(f"[warn] {label} 다운로드 실패: {e}")
        return rows

    for name, tkr in mapping.items():
        try:
            if len(mapping) == 1:
                close = data["Close"].dropna()
            else:
                close = data[tkr]["Close"].dropna()
            if len(close) < 2:
                continue
            last, prev = float(close.iloc[-1]), float(close.iloc[-2])
            chg = (last / prev - 1) * 100
            rows.append({
                "name": name,
                "last": last,
                "chg_pct": chg,
                "date": close.index[-1].strftime("%Y-%m-%d(%a)"),
            })
        except Exception:
            continue
    return rows


def fmt_rows(title, rows):
    if not rows:
        return f"[{title}]\n(데이터 없음)\n"
    out = [f"[{title}]"]
    for r in rows:
        out.append(f"- {r['name']}: {r['last']:,.2f} ({r['chg_pct']:+.2f}%) [{r['date']}]")
    return "\n".join(out) + "\n"


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
            title = entry.get("title", "").strip()
            if not title:
                continue
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_dt = dt.datetime(*published[:6], tzinfo=dt.timezone.utc)
                if pub_dt < cutoff:
                    continue
                stamp = pub_dt.strftime("%m-%d %H:%M UTC")
            else:
                stamp = ""
            summary = entry.get("summary", "")[:300].replace("\n", " ")
            items.append(f"- ({stamp}) {title} | {summary}")
    return "\n".join(items[:60]) if items else "(뉴스 수집 실패)"


def sector_movers(rows, n=3):
    if not rows:
        return ""
    s = sorted(rows, key=lambda r: r["chg_pct"], reverse=True)
    top = ", ".join(f"{r['name']} {r['chg_pct']:+.2f}%" for r in s[:n])
    bot = ", ".join(f"{r['name']} {r['chg_pct']:+.2f}%" for r in s[-n:])
    return f"[섹터 상위] {top}\n[섹터 하위] {bot}\n"


# ─────────────────────────────────────────────
# 원고 생성
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 한국 금융시장 실무자를 위한 데일리 글로벌 증시 브리핑을 쓰는 애널리스트입니다.

작성 원칙:
- 한국어로 작성. 문어체가 아닌 평이하고 직설적인 문장. 미사여구 금지.
- 데이터에 명시되지 않은 요일은 절대 추론해서 쓰지 말 것.
- 숫자는 반드시 제공된 데이터에서만 인용. 없는 숫자를 지어내지 말 것.
- 데이터에 없는 사실은 쓰지 말 것. 불확실하면 언급하지 않는 편이 낫다.
- 해석을 덧붙이되 단정하지 말 것. "~로 보인다", "~가 배경으로 지목된다" 수준.
- 투자 권유, 목표주가, 매수/매도 의견 금지.
- 네이버 블로그에 그대로 붙여넣을 형태로 출력. 마크다운 문법(##, **) 대신 일반 텍스트와 줄바꿈, 그리고 소제목은 그냥 한 줄로 쓸 것.

출력 구조:
1) 제목 한 줄
2) 세 줄 요약 (오늘 반드시 알아야 할 것)
3) 미국·유럽 지수 마감
4) 섹터별 등락과 특징주
5) 금리·환율·유가·금
6) 당일 한국시장 관전포인트
7) 맨 아래 한 줄: "※ 본 글은 공개 데이터를 정리한 것으로 투자 권유가 아닙니다."

분량은 전체 1,200~1,800자."""


def build_prompt(data_block, news_block, today):
    return f"""아래는 {today} 기준으로 수집한 원시 데이터입니다. 이를 바탕으로 블로그 원고를 작성해 주세요.

=== 시장 데이터 ===
{data_block}

=== 주요 뉴스 헤드라인 ===
{news_block}

=== 요청 ===
위 데이터만 사용해서 오늘자 브리핑 원고를 작성하세요. 데이터에 없는 수치나 사건은 절대 만들어내지 마세요."""


def generate_article(data_block, news_block, today):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(data_block, news_block, today)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# ─────────────────────────────────────────────
# 텔레그램 전송
# ─────────────────────────────────────────────
TG_LIMIT = 4000   # 텔레그램 메시지 최대 4096자. 여유를 둠


def split_text(text, limit=TG_LIMIT):
    """긴 텍스트를 줄 단위로 잘라 여러 조각으로"""
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
    """일반 텍스트로 전송 (마크다운 파싱 안 함 — 원고를 그대로 복사하기 위함)"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i, chunk in enumerate(split_text(text), 1):
        r = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }, timeout=30)
        if not r.ok:
            print(f"[error] 텔레그램 전송 실패 ({i}번째): {r.status_code} {r.text}")
            r.raise_for_status()
    print(f"[ok] 텔레그램 전송 완료")


def send_document(filename, content, caption=""):
    """원고를 .txt 파일로도 첨부 — 복사 붙여넣기 편하도록"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
            files={"document": (filename, content.encode("utf-8"), "text/plain")},
            timeout=60,
        )
        if not r.ok:
            print(f"[warn] 파일 첨부 실패: {r.text}")
    except Exception as e:
        print(f"[warn] 파일 첨부 실패: {e}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    check_config()

    kst = dt.timezone(dt.timedelta(hours=9))
    today = dt.datetime.now(kst).strftime("%Y-%m-%d (%a)")
    print(f"=== 글로벌 증시 브리핑 생성 시작: {today} KST ===")

    print("1/4 시장 데이터 수집 중...")
    idx = fetch_quotes(INDICES, "지수")
    sec = fetch_quotes(SECTORS, "섹터")
    wat = fetch_quotes(WATCHLIST, "특징주")
    mac = fetch_quotes(MACRO, "매크로")
    kor = fetch_quotes(KOSPI, "한국")

    data_block = "\n".join([
        fmt_rows("글로벌 주요 지수", idx),
        fmt_rows("미국 섹터 ETF", sec),
        sector_movers(sec),
        fmt_rows("주요 개별종목", wat),
        fmt_rows("금리·환율·원자재", mac),
        fmt_rows("한국 지수(전 거래일)", kor),
    ])

    print("2/4 뉴스 헤드라인 수집 중...")
    news_block = fetch_news()

    print("3/4 원고 생성 중...")
    article = generate_article(data_block, news_block, today)

    print("4/4 텔레그램 전송 중...")
    send_telegram(f"[글로벌 증시 브리핑] {today}\n\n{article}")
    send_document(
        f"brief_{dt.datetime.now(kst).strftime('%Y%m%d')}.txt",
        article + "\n\n" + "=" * 50 + "\n[수집 원본 데이터]\n" + "=" * 50 + "\n" + data_block,
        caption="원고 전문 + 원본 데이터 (숫자 대조용)",
    )

    os.makedirs("archive", exist_ok=True)
    fname = f"archive/brief_{dt.datetime.now(kst).strftime('%Y%m%d')}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(article + "\n\n" + data_block)

    print(f"완료. 아카이브: {fname}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            send_telegram(
                f"[브리핑 실패] {dt.datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"{traceback.format_exc()[-3000:]}"
            )
        except Exception:
            pass
        raise
