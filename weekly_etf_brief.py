# -*- coding: utf-8 -*-
"""
주간 ETF·테마 브리핑 (내부 참고용)
매주 월요일 07:00 KST 실행

수집:
  - 글로벌/한국 테마·섹터 ETF 1주/1개월/3개월 성과 랭킹
  - AUM 추정치 및 거래대금 변화 (자금흐름 근사)
  - 글로벌 ETF 업계 뉴스 (영문 RSS)
  - 한국 금융투자 기사 (구글뉴스 RSS 키워드 검색)

필요 Secrets: ANTHROPIC_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import urllib.parse
import datetime as dt

import requests
import yfinance as yf
import feedparser
import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

MODEL = "claude-sonnet-4-6"
KST = dt.timezone(dt.timedelta(hours=9))


def check_config():
    missing = [k for k in ("ANTHROPIC_API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID")
               if not os.environ.get(k)]
    if missing:
        print(f"[error] Secrets 미설정: {', '.join(missing)}")
        sys.exit(1)


# ═════════════════════════════════════════════
# 1. 수집 대상 ETF
# ═════════════════════════════════════════════

# 광의 섹터 (미국)
SECTOR_ETF = {
    "미국 기술": "XLK",
    "미국 금융": "XLF",
    "미국 헬스케어": "XLV",
    "미국 에너지": "XLE",
    "미국 산업재": "XLI",
    "미국 경기소비재": "XLY",
    "미국 필수소비재": "XLP",
    "미국 유틸리티": "XLU",
    "미국 소재": "XLB",
    "미국 부동산": "XLRE",
    "미국 커뮤니케이션": "XLC",
}

# 테마 ETF — 상품전략 관점에서 관찰 가치 있는 것들
THEME_ETF = {
    "AI/로보틱스(BOTZ)": "BOTZ",
    "AI 인프라(AIQ)": "AIQ",
    "반도체(SMH)": "SMH",
    "클라우드(SKYY)": "SKYY",
    "사이버보안(CIBR)": "CIBR",
    "혁신성장(ARKK)": "ARKK",
    "우주항공/방산(ITA)": "ITA",
    "글로벌 방산(SHLD)": "SHLD",
    "청정에너지(ICLN)": "ICLN",
    "원자력/우라늄(URA)": "URA",
    "2차전지/리튬(LIT)": "LIT",
    "전기차(DRIV)": "DRIV",
    "비만치료제(THNR)": "THNR",
    "바이오텍(XBI)": "XBI",
    "인프라(PAVE)": "PAVE",
    "귀금속채굴(GDX)": "GDX",
    "구리/광물(COPX)": "COPX",
    "농업(MOO)": "MOO",
    "블록체인(BLOK)": "BLOK",
    "핀테크(FINX)": "FINX",
    "게임/메타버스(ESPO)": "ESPO",
    "럭셔리소비(LUXE)": "LUXE",
    "고령화(AGNG)": "AGNG",
    "물/워터(PHO)": "PHO",
}

# 자산배분 / 지역 (플로우 판단용)
REGION_ETF = {
    "미국 대형(SPY)": "SPY",
    "미국 성장(IVW)": "IVW",
    "미국 가치(IVE)": "IVE",
    "미국 소형(IWM)": "IWM",
    "선진국 ex-US(EFA)": "EFA",
    "신흥국(EEM)": "EEM",
    "중국(MCHI)": "MCHI",
    "인도(INDA)": "INDA",
    "일본(EWJ)": "EWJ",
    "한국(EWY)": "EWY",
    "대만(EWT)": "EWT",
    "유럽(VGK)": "VGK",
    "미국채 장기(TLT)": "TLT",
    "미국채 단기(SHY)": "SHY",
    "투자등급회사채(LQD)": "LQD",
    "하이일드(HYG)": "HYG",
    "물가연동채(TIP)": "TIP",
    "금(GLD)": "GLD",
    "원자재(DBC)": "DBC",
    "비트코인 현물(IBIT)": "IBIT",
}

# 한국 상장 ETF (국내 시장 동향)
KR_ETF = {
    "KODEX 200": "069500.KS",
    "TIGER 미국S&P500": "360750.KS",
    "TIGER 미국나스닥100": "133690.KS",
    "KODEX 2차전지산업": "305720.KS",
    "TIGER 반도체": "091230.KS",
    "KODEX 은행": "091170.KS",
    "TIGER 미국배당다우존스": "458730.KS",
    "KODEX 종합채권액티브": "273130.KS",
    "TIGER 차이나전기차": "371460.KS",
    "KODEX 골드선물": "132030.KS",
}


# ═════════════════════════════════════════════
# 2. 성과 및 플로우 근사 수집
# ═════════════════════════════════════════════
def fetch_performance(mapping, label=""):
    """1주/1개월/3개월 수익률 + 거래대금 변화 수집"""
    rows = []
    tickers = list(mapping.values())
    try:
        data = yf.download(
            " ".join(tickers), period="4mo", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        print(f"[warn] {label} 실패: {e}")
        return rows

    for name, tkr in mapping.items():
        try:
            df = data[tkr] if len(tickers) > 1 else data
            close = df["Close"].dropna()
            vol = df["Volume"].dropna()
            if len(close) < 25:
                continue

            last = float(close.iloc[-1])

            def ret(n):
                if len(close) <= n:
                    return None
                return (last / float(close.iloc[-1 - n]) - 1) * 100

            r1w, r1m, r3m = ret(5), ret(21), ret(63)

            # 거래대금 근사: 최근 5일 평균 vs 직전 20일 평균
            dollar_vol = (close * vol).dropna()
            if len(dollar_vol) >= 25:
                recent = float(dollar_vol.iloc[-5:].mean())
                base = float(dollar_vol.iloc[-25:-5].mean())
                vol_chg = (recent / base - 1) * 100 if base > 0 else None
                avg_dv = recent
            else:
                vol_chg, avg_dv = None, None

            rows.append({
                "name": name, "ticker": tkr, "last": last,
                "r1w": r1w, "r1m": r1m, "r3m": r3m,
                "vol_chg": vol_chg, "avg_dv": avg_dv,
            })
        except Exception:
            continue
    return rows


def fetch_aum(mapping):
    """ETF AUM(순자산) 조회 — 규모 및 자금유입 판단 보조"""
    out = []
    for name, tkr in mapping.items():
        try:
            info = yf.Ticker(tkr).get_info()
            aum = info.get("totalAssets")
            if aum:
                out.append({"name": name, "aum": aum})
        except Exception:
            continue
        time.sleep(0.15)   # rate limit 회피
    return out


def fmt_perf(title, rows, sort_key="r1w", top_n=None):
    if not rows:
        return f"[{title}]\n(데이터 없음)\n"
    valid = [r for r in rows if r.get(sort_key) is not None]
    valid.sort(key=lambda r: r[sort_key], reverse=True)
    if top_n:
        valid = valid[:top_n]

    lines = [f"[{title}]", "종목 | 1주 | 1개월 | 3개월 | 거래대금변화(5d vs 20d)"]
    for r in valid:
        def p(v):
            return f"{v:+.1f}%" if v is not None else "n/a"
        vc = f"{r['vol_chg']:+.0f}%" if r["vol_chg"] is not None else "n/a"
        lines.append(f"- {r['name']}: {p(r['r1w'])} | {p(r['r1m'])} | {p(r['r3m'])} | 거래대금 {vc}")
    return "\n".join(lines) + "\n"


def fmt_extremes(title, rows, n=5):
    """상위/하위만 뽑아 요약"""
    valid = [r for r in rows if r.get("r1w") is not None]
    if not valid:
        return ""
    valid.sort(key=lambda r: r["r1w"], reverse=True)
    top = valid[:n]
    bot = valid[-n:]
    s = [f"[{title} — 주간 상위 {n}]"]
    for r in top:
        s.append(f"- {r['name']}: 주간 {r['r1w']:+.1f}%, 1개월 {r['r1m']:+.1f}%" if r['r1m'] is not None else f"- {r['name']}: 주간 {r['r1w']:+.1f}%")
    s.append(f"[{title} — 주간 하위 {n}]")
    for r in bot:
        s.append(f"- {r['name']}: 주간 {r['r1w']:+.1f}%, 1개월 {r['r1m']:+.1f}%" if r['r1m'] is not None else f"- {r['name']}: 주간 {r['r1w']:+.1f}%")
    return "\n".join(s) + "\n"


def fmt_aum(title, rows, top_n=15):
    if not rows:
        return ""
    rows = sorted(rows, key=lambda r: r["aum"], reverse=True)[:top_n]
    lines = [f"[{title}]"]
    for r in rows:
        aum_b = r["aum"] / 1e9
        lines.append(f"- {r['name']}: ${aum_b:,.1f}B")
    return "\n".join(lines) + "\n"


# ═════════════════════════════════════════════
# 3. 뉴스 수집
# ═════════════════════════════════════════════
GLOBAL_FEEDS = [
    ("ETF.com", "https://www.etf.com/rss.xml"),
    ("ETF Stream", "https://www.etfstream.com/feed/"),
    ("Reuters Funds", "https://news.google.com/rss/search?q=ETF+when:14d&hl=en-US&gl=US&ceid=US:en"),
    ("Global ETF Flows", "https://news.google.com/rss/search?q=%22ETF+flows%22+OR+%22fund+flows%22+when:14d&hl=en-US&gl=US&ceid=US:en"),
    ("ETF Launch", "https://news.google.com/rss/search?q=%22ETF+launch%22+OR+%22new+ETF%22+when:14d&hl=en-US&gl=US&ceid=US:en"),
    ("Thematic", "https://news.google.com/rss/search?q=%22thematic+ETF%22+OR+%22sector+rotation%22+when:14d&hl=en-US&gl=US&ceid=US:en"),
]

KR_QUERIES = [
    "ETF 순자산",
    "ETF 순유입",
    "ETF 신규상장",
    "상장지수펀드 시장",
    "자산운용사 ETF 경쟁",
    "퇴직연금 ETF",
    "펀드 설정액",
    "금융투자협회 펀드",
    "테마형 ETF",
]


def google_news_kr(query, when="14d"):
    q = urllib.parse.quote(f"{query} when:{when}")
    return f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def collect_feed(url, source, limit=10, days=21):
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    items = []
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"[warn] RSS 실패 {source}: {e}")
        return items

    for entry in feed.entries[:limit]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        pub = entry.get("published_parsed") or entry.get("updated_parsed")
        if pub:
            pdt = dt.datetime(*pub[:6], tzinfo=dt.timezone.utc)
            if pdt < cutoff:
                continue
            stamp = pdt.astimezone(KST).strftime("%m/%d")
        else:
            stamp = "--"
        summary = (entry.get("summary") or "")[:250].replace("\n", " ")
        items.append(f"- ({stamp}) {title} | {summary}")
    return items


def fetch_all_news():
    glob, kor = [], []

    for source, url in GLOBAL_FEEDS:
        glob += collect_feed(url, source, limit=10)

    for q in KR_QUERIES:
        kor += collect_feed(google_news_kr(q), q, limit=6)

    # 중복 제거 (제목 기준)
    def dedup(items):
        seen, out = set(), []
        for it in items:
            key = it.split("|")[0][-60:]
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    glob, kor = dedup(glob)[:45], dedup(kor)[:45]
    return (
        "\n".join(glob) or "(수집 실패)",
        "\n".join(kor) or "(수집 실패)",
    )


# ═════════════════════════════════════════════
# 4. 원고 생성
# ═════════════════════════════════════════════
SYSTEM_PROMPT = """당신은 ETF·펀드 상품전략 담당자를 위한 주간 시장 브리핑을 작성하는 애널리스트입니다.
독자는 자산운용사·연기금의 ETF 상품전략, 펀드 상품전략 실무자입니다. 시장을 이미 잘 아는 사람들이므로 기초 설명은 생략하고, 상품 기획과 라인업 판단에 쓸 수 있는 관찰에 집중하세요.

작성 원칙:
- 한국어. 평이하고 직설적인 문장. 미사여구·수식어 금지.
- 숫자는 제공된 데이터에서만 인용. 없는 숫자를 지어내지 말 것.
- 데이터에 없는 사실을 추론해서 쓰지 말 것. 특히 날짜, 요일, 자금유입 금액을 임의로 만들지 말 것.
- 거래대금 변화는 자금흐름의 '근사 지표'일 뿐이므로, 자금 유입·유출로 단정하지 말고 "거래 활발" "관심 증가" 수준으로 표현할 것.
- 해석은 덧붙이되 단정하지 말 것. "~로 보인다", "~가 배경으로 지목된다" 수준.
- 특정 종목·ETF에 대한 투자 권유, 목표가, 매수/매도 의견 금지.
- 뉴스는 제목과 요약만 주어지므로, 확인되지 않은 세부 내용을 지어내지 말 것. 불확실하면 언급하지 말 것.
- 텔레그램에서 읽으므로 마크다운 문법(##, **) 쓰지 말고 일반 텍스트와 줄바꿈으로 구성.

출력 구조:
1) 제목 한 줄
2) 이번 주 핵심 (3~5줄, 상품전략 관점에서 가장 중요한 것)
3) 테마·섹터 성과 랭킹
   - 주간·1개월 기준 상위/하위, 모멘텀 전환 구간(1개월과 1주 방향이 다른 것)을 짚을 것
4) 글로벌 ETF 시장 동향
   - 자금흐름 신호(거래대금 변화), 신규 상장, 업계 이슈
5) 지역·자산군 로테이션
6) 한국 시장 동향
7) 상품전략 관점 관찰 포인트
   - 라인업 공백, 해외에서 먼저 뜬 테마 중 국내 미출시 영역, 벤치마크·지수 수요 변화 등
   - 단정적 제안이 아니라 "관찰된 흐름"으로 서술
8) 맨 아래 한 줄: "※ 공개 데이터 기반 내부 참고용 자료. 투자 권유가 아니며, 거래대금 지표는 자금흐름의 근사치임."

분량은 2,500~3,500자."""


def build_prompt(blocks, today):
    return f"""아래는 {today} 기준으로 수집한 원시 데이터입니다. 주간 브리핑을 작성해 주세요.

{blocks}

=== 요청 ===
위 데이터만 사용해서 작성하세요. 데이터에 없는 수치, 자금유입액, 날짜, 사건은 절대 만들어내지 마세요.
뉴스는 제목 위주이므로 확인 가능한 범위에서만 언급하세요."""


def generate_article(blocks, today):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(blocks, today)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


# ═════════════════════════════════════════════
# 5. 텔레그램 전송
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
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }, timeout=30)
        if not r.ok:
            print(f"[error] 전송 실패({i}): {r.status_code} {r.text}")
            r.raise_for_status()
        time.sleep(0.5)
    print("[ok] 텔레그램 전송 완료")


def send_document(filename, content, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
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
    week_ago = (now - dt.timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"=== 주간 ETF 브리핑 시작: {today} ===")

    print("1/5 테마·섹터 성과 수집...")
    theme = fetch_performance(THEME_ETF, "테마")
    sector = fetch_performance(SECTOR_ETF, "섹터")
    region = fetch_performance(REGION_ETF, "지역/자산")
    kr = fetch_performance(KR_ETF, "한국")

    print("2/5 AUM 수집...")
    aum_theme = fetch_aum(THEME_ETF)

    print("3/5 뉴스 수집...")
    news_global, news_kr = fetch_all_news()

    print("4/5 원고 생성...")
    blocks = "\n".join([
        f"=== 수집 기준: {week_ago} ~ {now.strftime('%Y-%m-%d')} ===\n",
        "=== 테마 ETF 성과 (주간 수익률 순) ===",
        fmt_perf("테마 ETF 전체", theme, "r1w"),
        fmt_extremes("테마 ETF", theme, 5),
        "=== 섹터 ETF 성과 ===",
        fmt_perf("미국 섹터", sector, "r1w"),
        "=== 지역·자산군 ===",
        fmt_perf("지역/자산군", region, "r1w"),
        "=== 한국 상장 ETF ===",
        fmt_perf("한국 ETF", kr, "r1w"),
        "=== 테마 ETF 순자산 규모 ===",
        fmt_aum("AUM 상위", aum_theme),
        "=== 글로벌 ETF 관련 뉴스 (최근 2~3주) ===",
        news_global,
        "",
        "=== 한국 금융투자 관련 기사 (최근 2~3주) ===",
        news_kr,
    ])

    article = generate_article(blocks, today)

    print("5/5 전송...")
    send_telegram(f"[주간 ETF·테마 브리핑] {today}\n(내부 참고용)\n\n{article}")
    send_document(
        f"weekly_etf_{now.strftime('%Y%m%d')}.txt",
        article + "\n\n" + "=" * 60 + "\n[수집 원본 데이터]\n" + "=" * 60 + "\n" + blocks,
        caption="원고 + 원본 데이터 (숫자 대조용)",
    )

    os.makedirs("archive_weekly", exist_ok=True)
    fname = f"archive_weekly/weekly_{now.strftime('%Y%m%d')}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(article + "\n\n" + blocks)
    print(f"완료: {fname}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            send_telegram(f"[주간 브리핑 실패] {dt.datetime.now(KST):%Y-%m-%d}\n\n{traceback.format_exc()[-3000:]}")
        except Exception:
            pass
        raise
