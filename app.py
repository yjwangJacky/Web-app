import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(
    page_title="跨領域金融資訊展示平台",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# 教學用基礎資料
# -----------------------------
COMPANY_INFO = {
    "AAPL": {"name": "Apple", "sector": "Technology", "theme": "高市值科技股"},
    "MSFT": {"name": "Microsoft", "sector": "Technology", "theme": "雲端與AI"},
    "TSLA": {"name": "Tesla", "sector": "Automobile", "theme": "新能源與成長股"},
    "NVDA": {"name": "NVIDIA", "sector": "Semiconductor", "theme": "AI晶片"},
    "0050.TW": {"name": "元大台灣50", "sector": "ETF", "theme": "台灣大型權值股ETF"},
    "2330.TW": {"name": "台積電", "sector": "Semiconductor", "theme": "台灣半導體龍頭"},
}

BASE_ESG = {
    "AAPL": {"E": 74, "S": 78, "G": 81, "news_sentiment": 0.18},
    "MSFT": {"E": 81, "S": 84, "G": 87, "news_sentiment": 0.26},
    "TSLA": {"E": 69, "S": 60, "G": 63, "news_sentiment": -0.05},
    "NVDA": {"E": 72, "S": 76, "G": 79, "news_sentiment": 0.31},
    "0050.TW": {"E": 77, "S": 75, "G": 80, "news_sentiment": 0.12},
    "2330.TW": {"E": 83, "S": 79, "G": 86, "news_sentiment": 0.24},
}

TEACHING_NOTES = {
    "E": "環境面（Environment）：可用來說明碳排、能源效率、再生能源使用等議題。",
    "S": "社會面（Social）：可對應員工照顧、供應鏈責任、產品安全與消費者保護。",
    "G": "治理面（Governance）：可對應董事會結構、資訊透明度、內部控制與股東權益。",
}


# -----------------------------
# 工具函數
# -----------------------------
def score_to_grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 75:
        return "B"
    if score >= 65:
        return "C"
    return "D"


def compute_total_esg(esg_row: dict) -> float:
    return round(0.4 * esg_row["E"] + 0.3 * esg_row["S"] + 0.3 * esg_row["G"], 1)


def simulate_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """若無法抓到網路資料，使用可重現的教學示範資料。"""
    dates = pd.bdate_range(start, end)
    if len(dates) == 0:
        dates = pd.bdate_range(end=end, periods=120)

    seed = abs(hash(ticker)) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    drift = 0.00045 + (seed % 7) * 0.00003
    vol = 0.012 + (seed % 5) * 0.0015

    rets = rng.normal(loc=drift, scale=vol, size=len(dates))
    price = 100 * np.exp(np.cumsum(rets))

    df = pd.DataFrame({"Date": dates, "Close": price})
    df["Open"] = df["Close"].shift(1).fillna(df["Close"] * 0.995)
    intraday = rng.normal(0, vol * 0.45, size=len(df))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + np.abs(intraday))
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - np.abs(intraday))
    df["Volume"] = rng.integers(1_000_000, 5_000_000, size=len(df))
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(show_spinner=False)
def get_price_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    if yf is not None:
        try:
            df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            if not df.empty and {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                return df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        except Exception:
            pass
    return simulate_price_data(ticker, start, end)


@st.cache_data(show_spinner=False)
def build_universe_metrics(start: date, end: date) -> pd.DataFrame:
    rows = []
    for ticker, info in COMPANY_INFO.items():
        price_df = get_price_data(ticker, start, end)
        perf = calculate_metrics(price_df)
        esg = BASE_ESG[ticker].copy()
        esg_total = compute_total_esg(esg)
        rows.append(
            {
                "Ticker": ticker,
                "Company": info["name"],
                "Sector": info["sector"],
                "Theme": info["theme"],
                "E": esg["E"],
                "S": esg["S"],
                "G": esg["G"],
                "ESG Total": esg_total,
                "News Sentiment": esg["news_sentiment"],
                "Cumulative Return %": perf["cum_return_pct"],
                "Annualized Volatility %": perf["ann_vol_pct"],
                "Sharpe": perf["sharpe"],
                "Max Drawdown %": perf["max_drawdown_pct"],
            }
        )
    return pd.DataFrame(rows)


def calculate_metrics(df: pd.DataFrame) -> dict:
    data = df.copy()
    data["Return"] = data["Close"].pct_change()
    ret = data["Return"].dropna()
    if ret.empty:
        return {"cum_return_pct": 0.0, "ann_vol_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0}

    cum_return = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = 0 if ret.std() == 0 else ret.mean() / ret.std() * np.sqrt(252)
    wealth = (1 + ret).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    max_dd = drawdown.min()

    return {
        "cum_return_pct": round(cum_return * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
    }


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA60"] = data["Close"].rolling(60).mean()
    data["Return"] = data["Close"].pct_change()
    data["Vol20"] = data["Return"].rolling(20).std() * np.sqrt(252) * 100
    return data


# -----------------------------
# 側邊欄
# -----------------------------
st.sidebar.title("⚙️ 展示控制台")
st.sidebar.markdown("這是一個適合發表會的教學型金融資訊展示平台。")

selected_ticker = st.sidebar.selectbox("選擇展示標的", list(COMPANY_INFO.keys()), index=0)
end_date = st.sidebar.date_input("結束日期", value=date.today())
start_date = st.sidebar.date_input("開始日期", value=end_date - timedelta(days=365))
show_candle = st.sidebar.toggle("顯示K線圖", value=False)
show_teaching_box = st.sidebar.toggle("顯示教學解說", value=True)

if start_date >= end_date:
    st.sidebar.error("開始日期必須早於結束日期。")
    st.stop()

info = COMPANY_INFO[selected_ticker]
price_df = get_price_data(selected_ticker, start_date, end_date)
price_df = add_technical_indicators(price_df)
metrics = calculate_metrics(price_df)
esg = BASE_ESG[selected_ticker].copy()
esg_total = compute_total_esg(esg)
esg_grade = score_to_grade(esg_total)

# -----------------------------
# 頁首
# -----------------------------
st.title("📊 跨領域金融資訊顯示平台")
st.caption("教學展示版｜整合 ESG、股價、風險與情緒指標，適合課堂展示、成果發表與專題簡報。")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("展示標的", f"{selected_ticker}")
col_b.metric("累積報酬率", f"{metrics['cum_return_pct']:.2f}%")
col_c.metric("年化波動率", f"{metrics['ann_vol_pct']:.2f}%")
col_d.metric("ESG 總分", f"{esg_total:.1f}（{esg_grade}）")

st.markdown(
    f"**標的說明：** {info['name']}｜{info['sector']}｜{info['theme']}  "
    f"**資料期間：** {start_date} 至 {end_date}"
)

# -----------------------------
# 頁籤
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "一覽總表",
    "ESG 與跨領域分析",
    "風險與技術指標",
    "教學展示重點",
])

with tab1:
    left, right = st.columns([1.8, 1.2])

    with left:
        st.subheader("股價走勢")
        if show_candle:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=price_df["Date"],
                        open=price_df["Open"],
                        high=price_df["High"],
                        low=price_df["Low"],
                        close=price_df["Close"],
                        name="Price",
                    )
                ]
            )
            fig.update_layout(height=460, xaxis_title="Date", yaxis_title="Price")
        else:
            fig = px.line(price_df, x="Date", y="Close", title="收盤價走勢")
            fig.add_scatter(x=price_df["Date"], y=price_df["SMA20"], mode="lines", name="SMA20")
            fig.add_scatter(x=price_df["Date"], y=price_df["SMA60"], mode="lines", name="SMA60")
            fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("展示摘要")
        summary_df = pd.DataFrame(
            {
                "指標": ["累積報酬率", "年化波動率", "Sharpe Ratio", "最大回撤", "新聞情緒", "ESG 評等"],
                "數值": [
                    f"{metrics['cum_return_pct']:.2f}%",
                    f"{metrics['ann_vol_pct']:.2f}%",
                    f"{metrics['sharpe']:.2f}",
                    f"{metrics['max_drawdown_pct']:.2f}%",
                    f"{esg['news_sentiment']:.2f}",
                    esg_grade,
                ],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        latest_close = float(price_df["Close"].iloc[-1])
        first_close = float(price_df["Close"].iloc[0])
        st.info(
            f"本期間價格由 **{first_close:.2f}** 變動至 **{latest_close:.2f}**。"
            f" 若搭配 ESG 與情緒分數，可以延伸討論『永續表現是否與市場評價同步』。"
        )

    st.subheader("跨標的比較")
    universe_df = build_universe_metrics(start_date, end_date)
    st.dataframe(universe_df, use_container_width=True, hide_index=True)

with tab2:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("E / S / G 分項雷達圖")
        radar_df = pd.DataFrame(
            {
                "Category": ["E", "S", "G", "E"],
                "Score": [esg["E"], esg["S"], esg["G"], esg["E"]],
            }
        )
        radar = px.line_polar(radar_df, r="Score", theta="Category", line_close=True)
        radar.update_traces(fill="toself")
        radar.update_layout(height=420)
        st.plotly_chart(radar, use_container_width=True)

    with c2:
        st.subheader("ESG 組成")
        bar_df = pd.DataFrame({"Dimension": ["E", "S", "G"], "Score": [esg["E"], esg["S"], esg["G"]]})
        bar = px.bar(bar_df, x="Dimension", y="Score", text="Score", title="ESG 分項分數")
        bar.update_layout(height=420)
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("ESG 與報酬率散佈圖")
    scatter_df = build_universe_metrics(start_date, end_date)
    scatter = px.scatter(
        scatter_df,
        x="ESG Total",
        y="Cumulative Return %",
        size="Annualized Volatility %",
        color="Sector",
        hover_name="Ticker",
        text="Ticker",
        title="不同標的的 ESG 與報酬率關係",
    )
    scatter.update_traces(textposition="top center")
    st.plotly_chart(scatter, use_container_width=True)

    st.subheader("新聞情緒 vs. ESG 總分")
    sent = px.scatter(
        scatter_df,
        x="News Sentiment",
        y="ESG Total",
        color="Sector",
        text="Ticker",
        hover_name="Company",
        title="情緒分數與 ESG 的示意關係",
    )
    sent.update_traces(textposition="top center")
    st.plotly_chart(sent, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("滾動波動率")
        vol_fig = px.line(price_df, x="Date", y="Vol20", title="20日年化波動率（%）")
        vol_fig.update_layout(height=400)
        st.plotly_chart(vol_fig, use_container_width=True)

    with c2:
        st.subheader("日報酬率分布")
        hist = px.histogram(price_df.dropna(), x="Return", nbins=40, title="日報酬率直方圖")
        hist.update_layout(height=400)
        st.plotly_chart(hist, use_container_width=True)

    st.subheader("風險指標解讀")
    risk_table = pd.DataFrame(
        {
            "指標": ["Sharpe Ratio", "最大回撤", "年化波動率"],
            "教學解讀": [
                "報酬相對風險的效率指標，數值越高代表單位風險帶來的報酬越高。",
                "衡量歷史上從高點回落的最深幅度，可用來說明下行風險。",
                "衡量報酬變動程度，數值越大代表價格越不穩定。",
            ],
            "本標的數值": [
                f"{metrics['sharpe']:.2f}",
                f"{metrics['max_drawdown_pct']:.2f}%",
                f"{metrics['ann_vol_pct']:.2f}%",
            ],
        }
    )
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("  ")
    st.markdown(
        """
### 1. 問題意識
投資人在面對股價、ESG、新聞與風險時，常常要切換不同網站與資料來源，因此不利於快速做出判斷。

### 2. 平台特色
本平台將財務資料、ESG 指標、情緒訊號與風險分析整合到同一個畫面，適合教學展示與跨領域成果發表。

### 3. 跨領域價值
除了傳統股價資訊外，平台也納入永續評分與新聞情緒，讓使用者理解金融決策不只看報酬，也要看永續與資訊環境。

### 4. 展示亮點
可以即時切換標的、比較不同資產、說明 ESG 與報酬的關係，並搭配風險圖表進行完整展示。
        """
    )

    if show_teaching_box:
        st.subheader("ESG 教學說明框")
        for k, v in TEACHING_NOTES.items():
            st.success(f"**{k}：** {v}")

    st.subheader(" 3 分鐘了解此平台")
    demo_script = pd.DataFrame(
        {
            "時間": ["0:00–0:30", "0:30–1:20", "1:20–2:10", "2:10–3:00"],
            "內容": [
                "說明投資資訊分散、ESG 與市場資訊難整合的問題。",
                "展示單一股票的價格走勢、均線與基本風險指標。",
                "切換到 ESG 頁面，展示 E/S/G 分數與 ESG vs 報酬散佈圖。",
                "本平台可作為教學、研究展示與投資決策輔助工具。",
            ],
        }
    )
    st.dataframe(demo_script, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "備註：本平台預設優先嘗試抓取 Yahoo Finance 市價資料；若執行環境無法連網，會自動切換為教學示範資料。"
)
