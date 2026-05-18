import time
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="ESG-AI Smart Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
DATA_DIR = Path("data")
STOCK_PRICE_FILE = DATA_DIR / "stock_price.csv"
ESG_FILE = DATA_DIR / "esg_scores.csv"
FINANCIAL_FILE = DATA_DIR / "financials.csv"
ETF_FILE = DATA_DIR / "etf_prices.csv"

DEFAULT_TOP_N = 10
DEFAULT_RF_TREES = 200
TRADING_DAYS = 252
RISK_FREE_RATE = 0.01
DEFAULT_ETFS = ["0050.TW", "0056.TW", "00878.TW", "009809.TW"]


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"讀取檔案失敗：{path.name}，原因：{e}")
            return None
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_ticker_for_merge(ticker: str) -> str:
    if pd.isna(ticker):
        return ticker
    ticker = str(ticker).strip().upper()
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return ticker
    if ticker.isdigit() and len(ticker) == 4:
        return f"{ticker}.TW"
    return ticker


def infer_price_wide_or_long(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_datetime(df, date_col)

    if {"date", "ticker", "close"}.issubset(df.columns):
        df = df[["date", "ticker", "close"]].dropna()
        df["ticker"] = df["ticker"].astype(str).apply(normalize_ticker_for_merge)
        return df

    if date_col in df.columns:
        value_cols = [c for c in df.columns if c != date_col]
        long_df = df.melt(
            id_vars=[date_col],
            value_vars=value_cols,
            var_name="ticker",
            value_name="close"
        )
        long_df = long_df.rename(columns={date_col: "date"}).dropna()
        long_df["ticker"] = long_df["ticker"].astype(str).apply(normalize_ticker_for_merge)
        return long_df

    raise ValueError("價格資料必須包含 long format(date,ticker,close) 或 wide format(date, ticker1, ticker2, ...) 格式")


def preprocess_esg(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    rename_map = {
        "stock_id": "ticker",
        "code": "ticker",
        "symbol": "ticker",
        "yearmonth": "date",
        "month": "date",
        "esg": "esg_total",
        "e": "e_score",
        "s": "s_score",
        "g": "g_score",
        "controversy": "controversy_score",
        "carbon": "carbon_intensity",
        "carbon_emission": "carbon_intensity",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "ticker" not in df.columns:
        raise ValueError("ESG 資料必須包含 ticker 欄位")

    if "date" in df.columns:
        df = ensure_datetime(df, "date")
    else:
        df["date"] = pd.NaT

    df["ticker"] = df["ticker"].astype(str).apply(normalize_ticker_for_merge)
    return df


def preprocess_financials(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    rename_map = {
        "stock_id": "ticker",
        "code": "ticker",
        "symbol": "ticker",
        "yearmonth": "date",
        "month": "date",
        "pb": "pb",
        "pe": "pe",
        "debt_ratio": "debt_ratio",
        "lev": "debt_ratio",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "ticker" not in df.columns:
        raise ValueError("財務資料必須包含 ticker 欄位")

    if "date" in df.columns:
        df = ensure_datetime(df, "date")
    else:
        df["date"] = pd.NaT

    df["ticker"] = df["ticker"].astype(str).apply(normalize_ticker_for_merge)
    return df


def yahoo_ticker_format(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return ticker
    if ticker.isdigit() and len(ticker) == 4:
        return f"{ticker}.TW"
    return ticker


def fetch_yahoo_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise ImportError("尚未安裝 yfinance。請先 pip install yfinance")

    tickers = [yahoo_ticker_format(t) for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("請至少輸入一個 ticker")

    all_rows = []
    failed = []
    progress = st.progress(0, text="開始從 Yahoo Finance 載入資料...")

    for idx, ticker in enumerate(tickers, start=1):
        success = False

        for attempt in range(3):
            try:
                df = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    group_by="column",
                )

                if df is None or df.empty:
                    raise ValueError(f"{ticker} 沒有抓到資料")

                if "Close" in df.columns:
                    temp = df[["Close"]].reset_index().rename(
                        columns={"Date": "date", "Close": "close"}
                    )
                else:
                    close_candidates = []
                    for c in df.columns:
                        c_str = str(c)
                        if c_str == "Close" or "Close" in c_str:
                            close_candidates.append(c)

                    if not close_candidates:
                        raise ValueError(f"{ticker} 找不到 Close 欄位")

                    temp = df[[close_candidates[0]]].reset_index()
                    temp.columns = ["date", "close"]

                temp["ticker"] = ticker
                temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
                temp["close"] = pd.to_numeric(temp["close"], errors="coerce")
                temp = temp.dropna(subset=["date", "close"])

                if temp.empty:
                    raise ValueError(f"{ticker} 抓回資料後為空")

                all_rows.append(temp[["date", "ticker", "close"]])
                success = True
                break

            except Exception:
                wait_sec = 2 + attempt * 3 + random.uniform(0.5, 1.5)
                time.sleep(wait_sec)

        if not success:
            failed.append(ticker)

        progress.progress(
            idx / len(tickers),
            text=f"Yahoo Finance 載入中... {idx}/{len(tickers)}"
        )

        time.sleep(1.5 + random.uniform(0.2, 0.8))

    progress.empty()

    if failed:
        st.warning(f"以下 ticker 從 Yahoo Finance 抓取失敗：{', '.join(failed)}")

    if not all_rows:
        raise ValueError("Yahoo Finance 沒有回傳任何可用價格資料，請稍後再試，或改用本機 CSV。")

    out = pd.concat(all_rows, ignore_index=True)
    out["ticker"] = out["ticker"].astype(str).apply(normalize_ticker_for_merge)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def compute_daily_returns(price_long: pd.DataFrame) -> pd.DataFrame:
    df = price_long.copy().sort_values(["ticker", "date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    短樣本可跑版：
    - 技術指標視窗縮短為 3 / 5
    - 預測目標改為未來 5 日報酬
    """
    df = df.copy().sort_values(["ticker", "date"])
    g = df.groupby("ticker")

    df["mom_3"] = g["close"].pct_change(3)
    df["mom_5"] = g["close"].pct_change(5)
    df["vol_3"] = g["ret_1d"].rolling(3).std().reset_index(level=0, drop=True) * np.sqrt(TRADING_DAYS)
    df["vol_5"] = g["ret_1d"].rolling(5).std().reset_index(level=0, drop=True) * np.sqrt(TRADING_DAYS)
    df["avg_close_5"] = g["close"].rolling(5).mean().reset_index(level=0, drop=True)

    # 短樣本預測目標
    df["fwd_ret_5"] = g["close"].shift(-5) / df["close"] - 1

    return df


def merge_features(price_df: pd.DataFrame, esg_df: Optional[pd.DataFrame], fin_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    base = add_technical_features(compute_daily_returns(price_df))
    base["year_month"] = base["date"].dt.to_period("M").astype(str)

    if esg_df is not None and not esg_df.empty:
        esg_df = esg_df.copy()
        if esg_df["date"].isna().all():
            base = base.merge(esg_df.drop(columns=["date"]), on="ticker", how="left")
        else:
            esg_df["year_month"] = esg_df["date"].dt.to_period("M").astype(str)
            esg_monthly = esg_df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "year_month"], keep="last")
            base = base.merge(esg_monthly.drop(columns=["date"]), on=["ticker", "year_month"], how="left")

    if fin_df is not None and not fin_df.empty:
        fin_df = fin_df.copy()
        if fin_df["date"].isna().all():
            base = base.merge(fin_df.drop(columns=["date"]), on="ticker", how="left")
        else:
            fin_df["year_month"] = fin_df["date"].dt.to_period("M").astype(str)
            fin_monthly = fin_df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "year_month"], keep="last")
            base = base.merge(fin_monthly.drop(columns=["date"]), on=["ticker", "year_month"], how="left")

    return base


def apply_esg_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "controversy_score" in out.columns:
        out = out[(out["controversy_score"].isna()) | (out["controversy_score"] <= 3)]

    if "eps" in out.columns:
        out = out[(out["eps"].isna()) | (out["eps"] >= 0)]

    if "esg_total" in out.columns:
        med = out.groupby("date")["esg_total"].transform("median")
        out = out[(out["esg_total"].isna()) | (out["esg_total"] >= med)]

    return out


def build_rule_based_portfolio(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    data = apply_esg_rules(df)

    score_cols = [c for c in [
        "esg_total", "e_score", "s_score", "g_score",
        "roe", "roa", "mom_3", "mom_5"
    ] if c in data.columns]

    if not score_cols:
        raise ValueError("缺少可用於規則式 ESG 選股的欄位")

    rank_df = data[["date", "ticker", "close", "fwd_ret_5"] + score_cols].copy()

    for c in score_cols:
        rank_df[f"rank_{c}"] = rank_df.groupby("date")[c].rank(pct=True)

    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    rank_df["rule_score"] = rank_df[rank_cols].mean(axis=1)

    selected = (
        rank_df.sort_values(["date", "rule_score"], ascending=[True, False])
        .groupby("date")
        .head(top_n)
        .copy()
    )
    selected["weight"] = 1 / top_n
    selected["portfolio"] = "Rule-based ESG"
    return selected


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "ticker", "close", "ret_1d", "fwd_ret_5", "year_month"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def fill_feature_na_with_median(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and not df[col].isna().all():
            df[col] = df[col].fillna(df[col].median())
    return df


def build_ai_portfolio(df: pd.DataFrame, model_name: str, top_n: int, n_estimators: int):
    data = apply_esg_rules(df).dropna(subset=["fwd_ret_5"]).copy()
    feature_cols = get_feature_columns(data)

    if not feature_cols:
        raise ValueError("缺少可用於 AI 模型的特徵")

    data = fill_feature_na_with_median(data, feature_cols)
    data = data.dropna(subset=["fwd_ret_5"]).copy()

    if data.empty:
        raise ValueError("特徵資料在清理缺漏值後為空，請補充 ESG/財務資料或縮短特徵需求。")

    split_date = data["date"].quantile(0.8)
    train_df = data[data["date"] <= split_date].copy()
    test_df = data[data["date"] > split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("訓練集或測試集為空，請調整資料期間。")

    X_train = train_df[feature_cols]
    y_train = train_df["fwd_ret_5"]
    X_test = test_df[feature_cols]
    y_test = test_df["fwd_ret_5"]

    if model_name == "Random Forest":
        if not SKLEARN_AVAILABLE:
            raise ImportError("請安裝 scikit-learn")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("請安裝 xgboost")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        raise ValueError("不支援的模型")

    model.fit(X_train, y_train)

    eval_df = pd.DataFrame({
        "date": test_df["date"],
        "ticker": test_df["ticker"],
        "y_true": y_test,
        "y_pred": model.predict(X_test),
    })

    selected = test_df[["date", "ticker", "close", "fwd_ret_5"] + feature_cols].copy()
    selected["pred_score"] = model.predict(selected[feature_cols])
    selected = (
        selected.sort_values(["date", "pred_score"], ascending=[True, False])
        .groupby("date")
        .head(top_n)
        .copy()
    )
    selected["weight"] = 1 / top_n
    selected["portfolio"] = f"AI-ESG ({model_name})"

    return selected, model, feature_cols, eval_df


def compute_portfolio_returns(selected_df: pd.DataFrame) -> pd.DataFrame:
    if selected_df is None or selected_df.empty:
        return pd.DataFrame(columns=["date", "ret", "cum"])

    df = selected_df.copy()
    df["weighted_ret"] = df["weight"] * df["fwd_ret_5"]

    out = (
        df.groupby("date", as_index=False)["weighted_ret"]
        .sum()
        .rename(columns={"weighted_ret": "ret"})
    )
    out["cum"] = (1 + out["ret"].fillna(0)).cumprod()
    return out


def compute_etf_returns(etf_long: pd.DataFrame) -> pd.DataFrame:
    if etf_long is None or etf_long.empty:
        return pd.DataFrame(columns=["date", "portfolio", "ret", "cum"])

    rows = []
    for ticker, g in etf_long.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["ret"] = g["close"].pct_change()
        g["cum"] = (1 + g["ret"].fillna(0)).cumprod()
        g["portfolio"] = ticker.upper()
        rows.append(g[["date", "portfolio", "ret", "cum"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "portfolio", "ret", "cum"])


def performance_summary(return_df: pd.DataFrame, ret_col: str = "ret") -> dict:
    r = return_df[ret_col].dropna()

    if len(r) == 0:
        return {
            "Cumulative Return": np.nan,
            "Annual Return": np.nan,
            "Annual Volatility": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown": np.nan,
        }

    cum = (1 + r).prod() - 1
    ann_ret = (1 + r.mean()) ** TRADING_DAYS - 1
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    return {
        "Cumulative Return": cum,
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown.min(),
    }


def fmt_pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x: float) -> str:
    return "-" if pd.isna(x) else f"{x:.3f}"


@st.cache_data(show_spinner=False)
def load_local_data():
    stock_raw = safe_read_csv(STOCK_PRICE_FILE)
    esg_raw = safe_read_csv(ESG_FILE)
    fin_raw = safe_read_csv(FINANCIAL_FILE)
    etf_raw = safe_read_csv(ETF_FILE)

    stock = infer_price_wide_or_long(stock_raw) if stock_raw is not None else None
    esg = preprocess_esg(esg_raw) if esg_raw is not None else None
    fin = preprocess_financials(fin_raw) if fin_raw is not None else None
    etf = infer_price_wide_or_long(etf_raw) if etf_raw is not None else None

    return stock, esg, fin, etf


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("📊 ESG-AI Smart Portfolio Dashboard")
st.caption("AI 選股 × ESG 因子 × ESG ETF 比較平台（短樣本可跑版）")

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 參數設定")
    data_mode = st.radio("資料來源模式", ["本機 CSV", "Yahoo Finance 自動下載"])
    model_name = st.selectbox("AI 模型", ["Random Forest", "XGBoost"])
    top_n = st.slider("Top N 選股數", 5, 20, DEFAULT_TOP_N, 1)
    n_estimators = st.slider("樹的數量", 50, 500, DEFAULT_RF_TREES, 50)

    st.markdown("---")
    st.markdown("**ETF 對照清單**")
    compare_etfs = st.multiselect(
        "比較 ETF",
        DEFAULT_ETFS,
        default=DEFAULT_ETFS
    )

    st.markdown("---")
    if data_mode == "Yahoo Finance 自動下載":
        st.caption("Yahoo Finance 容易限流，建議一次先抓 3–5 檔股票；展示時建議優先使用本機 CSV。")
        stock_input = st.text_area(
            "股票 ticker（逗號分隔）",
            value="2330.TW,2317.TW,2454.TW"
        )
        start_str = st.text_input("開始日期", value="2024-01-01")
        end_str = st.text_input("結束日期", value=pd.Timestamp.today().strftime("%Y-%m-%d"))
        load_btn = st.button("從 Yahoo Finance 載入資料")
    else:
        st.code("""data/stock_price.csv
data/esg_scores.csv
data/financials.csv
data/etf_prices.csv""")
        load_btn = st.button("載入本機資料")


# ------------------------------------------------------------
# Data loading logic
# ------------------------------------------------------------
stock_price = None
esg_scores = None
financials = None
etf_prices = None

if data_mode == "本機 CSV":
    if load_btn:
        try:
            stock_price, esg_scores, financials, etf_prices = load_local_data()

            st.session_state["loaded"] = True
            st.session_state["stock_price"] = stock_price
            st.session_state["esg_scores"] = esg_scores
            st.session_state["financials"] = financials
            st.session_state["etf_prices"] = etf_prices

            st.success("本機資料載入成功。")
        except Exception as e:
            st.error(f"本機資料載入失敗：{e}")

else:
    if load_btn:
        try:
            tickers = [x.strip() for x in stock_input.split(",") if x.strip()]

            stock_price = fetch_yahoo_prices(tickers, start_str, end_str)

            etf_prices = None
            if compare_etfs:
                try:
                    etf_prices = fetch_yahoo_prices(compare_etfs, start_str, end_str)
                except Exception as e:
                    st.warning(f"ETF 價格抓取失敗：{e}")
                    etf_prices = None

            local_esg_raw = safe_read_csv(ESG_FILE)
            local_fin_raw = safe_read_csv(FINANCIAL_FILE)

            esg_scores = preprocess_esg(local_esg_raw) if local_esg_raw is not None else None
            financials = preprocess_financials(local_fin_raw) if local_fin_raw is not None else None

            if esg_scores is None:
                st.info("未找到本機 esg_scores.csv，系統將只使用價格與財務/技術因子。")

            if financials is None:
                st.info("未找到本機 financials.csv，系統將只使用價格與 ESG/技術因子。")

            st.session_state["loaded"] = True
            st.session_state["stock_price"] = stock_price
            st.session_state["esg_scores"] = esg_scores
            st.session_state["financials"] = financials
            st.session_state["etf_prices"] = etf_prices

            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                stock_price.to_csv(STOCK_PRICE_FILE, index=False, encoding="utf-8-sig")
                if etf_prices is not None and not etf_prices.empty:
                    etf_prices.to_csv(ETF_FILE, index=False, encoding="utf-8-sig")
                st.success("Yahoo Finance 價格資料已載入，並已快取到 data/ 資料夾。")
            except Exception as cache_err:
                st.warning(f"資料已載入，但快取存檔失敗：{cache_err}")

        except Exception as e:
            st.error(f"Yahoo Finance 載入失敗：{e}")
            st.info("建議改用本機 CSV，或減少 ticker 數量後再試。")


if st.session_state.get("loaded", False):
    stock_price = st.session_state.get("stock_price")
    esg_scores = st.session_state.get("esg_scores")
    financials = st.session_state.get("financials")
    etf_prices = st.session_state.get("etf_prices")


if stock_price is None or stock_price.empty:
    st.info("請先在左側選擇資料來源並載入資料。")
    st.stop()

# ------------------------------------------------------------
# Raw data downloads
# ------------------------------------------------------------
st.markdown("### 原始資料下載")
col_d1, col_d2, col_d3, col_d4 = st.columns(4)

with col_d1:
    st.download_button(
        "下載股票價格 CSV",
        to_csv_download(stock_price),
        file_name="stock_price_export.csv",
        mime="text/csv"
    )

with col_d2:
    if esg_scores is not None and not esg_scores.empty:
        st.download_button(
            "下載 ESG CSV",
            to_csv_download(esg_scores),
            file_name="esg_scores_export.csv",
            mime="text/csv"
        )

with col_d3:
    if financials is not None and not financials.empty:
        st.download_button(
            "下載財務 CSV",
            to_csv_download(financials),
            file_name="financials_export.csv",
            mime="text/csv"
        )

with col_d4:
    if etf_prices is not None and not etf_prices.empty:
        st.download_button(
            "下載 ETF CSV",
            to_csv_download(etf_prices),
            file_name="etf_prices_export.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
features_df = merge_features(stock_price, esg_scores, financials)

if features_df is None or features_df.empty:
    st.error("合併後的特徵資料為空，請檢查股票價格、ESG 與財務資料是否成功載入。")
    st.stop()

if "date" not in features_df.columns:
    st.error("features_df 中找不到 date 欄位，請檢查資料格式。")
    st.stop()

features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")
features_df = features_df.dropna(subset=["date"]).copy()

if features_df.empty:
    st.error("有效日期資料為空，請檢查 stock_price / ESG / financials 格式。")
    st.stop()

features_df = features_df.sort_values(["date", "ticker"]).copy()

min_date = features_df["date"].min()
max_date = features_df["date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("日期範圍無法判定，請檢查資料中的 date 欄位。")
    st.stop()

selected_range = st.sidebar.date_input(
    "回測期間",
    value=(min_date.date(), max_date.date())
)

start_date = pd.to_datetime(selected_range[0])
end_date = pd.to_datetime(selected_range[1])

features_df = features_df[
    (features_df["date"] >= start_date) &
    (features_df["date"] <= end_date)
].copy()

if features_df.empty:
    st.warning("所選回測期間內沒有資料。")
    st.stop()

# Build portfolios
try:
    rule_port = build_rule_based_portfolio(features_df, top_n=top_n)
    ai_port, trained_model, feature_cols, eval_df = build_ai_portfolio(
        features_df, model_name, top_n, n_estimators
    )
except Exception as e:
    st.error(f"投組建構失敗：{e}")
    st.stop()

rule_ret = compute_portfolio_returns(rule_port)
rule_ret["portfolio"] = "Rule-based ESG"

ai_ret = compute_portfolio_returns(ai_port)
ai_ret["portfolio"] = f"AI-ESG ({model_name})"

portfolio_panel = pd.concat([rule_ret, ai_ret], ignore_index=True)

etf_panel = pd.DataFrame(columns=["date", "portfolio", "ret", "cum"])
if etf_prices is not None and not etf_prices.empty:
    etf_prices = etf_prices.copy()
    etf_prices["date"] = pd.to_datetime(etf_prices["date"], errors="coerce")
    etf_prices = etf_prices.dropna(subset=["date"])
    etf_prices = etf_prices[
        (etf_prices["date"] >= start_date) &
        (etf_prices["date"] <= end_date)
    ].copy()

    if not etf_prices.empty:
        etf_prices["ticker"] = etf_prices["ticker"].astype(str).apply(normalize_ticker_for_merge)
        etf_panel = compute_etf_returns(etf_prices)

all_series = pd.concat([portfolio_panel, etf_panel], ignore_index=True)

if all_series.empty:
    st.error("沒有可用的策略或 ETF 報酬資料。")
    st.stop()

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["首頁總覽", "資料與選股邏輯", "AI 選股結果", "績效比較", "市場情境分析", "模型解釋"]
)

# ------------------------------------------------------------
# Tab 1
# ------------------------------------------------------------
with tab1:
    st.subheader("專題總覽")

    ai_summary = performance_summary(ai_ret)
    rule_summary = performance_summary(rule_ret)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI-ESG 年化報酬", fmt_pct(ai_summary["Annual Return"]))
    c2.metric("AI-ESG Sharpe", fmt_num(ai_summary["Sharpe"]))
    c3.metric("AI-ESG 最大回撤", fmt_pct(ai_summary["Max Drawdown"]))
    c4.metric("Rule-based 年化報酬", fmt_pct(rule_summary["Annual Return"]))

    fig = px.line(
        all_series.sort_values("date"),
        x="date",
        y="cum",
        color="portfolio",
        title="累積報酬比較"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Tab 2
# ------------------------------------------------------------
with tab2:
    st.subheader("資料與選股邏輯")
    st.write(f"目前模式：{data_mode}")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 研究流程")
        st.markdown(
            "原始股票池 → ESG/財務過濾 → 特徵工程 → AI 模型評分 → Top N 選股 → 投組回測 → ETF 比較"
        )

        st.markdown("#### Rule-based ESG 邏輯")
        rules = []
        if "controversy_score" in features_df.columns:
            rules.append("爭議事件分數過高者排除")
        if "eps" in features_df.columns:
            rules.append("EPS 為負者排除")
        if "esg_total" in features_df.columns:
            rules.append("ESG 總分低於當期中位數者排除")
        st.write(rules if rules else ["目前根據可用欄位進行過濾與排序"])

    with c2:
        feat_df = pd.DataFrame({"feature": get_feature_columns(features_df)})
        st.dataframe(feat_df, use_container_width=True, height=320)

# ------------------------------------------------------------
# Tab 3
# ------------------------------------------------------------
with tab3:
    st.subheader("AI 選股結果")

    dates = sorted(ai_port["date"].dropna().unique())
    if dates:
        selected_date = st.selectbox(
            "選擇日期",
            dates,
            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
        )

        picks = ai_port[ai_port["date"] == selected_date].sort_values(
            "pred_score", ascending=False
        ).copy()

        show_cols = [c for c in [
            "ticker", "pred_score", "weight",
            "esg_total", "e_score", "s_score", "g_score",
            "roe", "roa", "mom_3", "mom_5", "vol_3", "vol_5"
        ] if c in picks.columns]

        st.dataframe(picks[show_cols], use_container_width=True, height=420)

        st.download_button(
            "下載當期 AI 選股結果",
            to_csv_download(picks),
            file_name="ai_picks_selected_date.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------
# Tab 4
# ------------------------------------------------------------
with tab4:
    st.subheader("績效比較")

    rows = []
    for name, sub in all_series.groupby("portfolio"):
        s = performance_summary(sub)
        s["Portfolio"] = name
        rows.append(s)

    perf_table = pd.DataFrame(rows)[[
        "Portfolio", "Cumulative Return", "Annual Return",
        "Annual Volatility", "Sharpe", "Max Drawdown"
    ]]

    st.dataframe(
        perf_table.style.format({
            "Cumulative Return": "{:.2%}",
            "Annual Return": "{:.2%}",
            "Annual Volatility": "{:.2%}",
            "Sharpe": "{:.3f}",
            "Max Drawdown": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.download_button(
        "下載績效比較表",
        to_csv_download(perf_table),
        file_name="performance_comparison.csv",
        mime="text/csv"
    )

    fig = px.scatter(
        perf_table,
        x="Annual Volatility",
        y="Annual Return",
        text="Portfolio",
        title="風險報酬散點圖"
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Tab 5
# ------------------------------------------------------------
with tab5:
    st.subheader("市場情境分析")

    regime_options = {
        "全部期間": (start_date, end_date),
        "前半段": (start_date, start_date + (end_date - start_date) / 2),
        "後半段": (start_date + (end_date - start_date) / 2, end_date),
    }

    regime_name = st.selectbox("市場情境", list(regime_options.keys()))
    r_start, r_end = regime_options[regime_name]

    regime_df = all_series[
        (all_series["date"] >= r_start) & (all_series["date"] <= r_end)
    ].copy()

    if regime_df.empty:
        st.info("此情境區間沒有可用資料。")
    else:
        regime_df["cum_regime"] = regime_df.groupby("portfolio")["ret"].transform(
            lambda x: (1 + x.fillna(0)).cumprod()
        )
        fig = px.line(
            regime_df,
            x="date",
            y="cum_regime",
            color="portfolio",
            title=f"{regime_name} 累積報酬"
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Tab 6
# ------------------------------------------------------------
with tab6:
    st.subheader("模型解釋")

    if hasattr(trained_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": trained_model.feature_importances_
        }).sort_values("importance", ascending=False)

        fig = px.bar(
            importance_df.head(15),
            x="feature",
            y="importance",
            title="特徵重要性 Top 15"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "下載特徵重要性",
            to_csv_download(importance_df),
            file_name="feature_importance.csv",
            mime="text/csv"
        )

    if eval_df is not None and not eval_df.empty and SKLEARN_AVAILABLE:
        rmse = mean_squared_error(eval_df["y_true"], eval_df["y_pred"], squared=False)
        st.metric("測試集 RMSE", fmt_num(rmse))

        st.download_button(
            "下載模型預測結果",
            to_csv_download(eval_df),
            file_name="model_predictions.csv",
            mime="text/csv"
        )

    if SHAP_AVAILABLE and hasattr(trained_model, "predict"):
        with st.expander("SHAP 分析（可能較慢）"):
            try:
                sample_df = apply_esg_rules(features_df).dropna(subset=["fwd_ret_5"]).copy()
                sample_df = fill_feature_na_with_median(sample_df, get_feature_columns(sample_df))
                sample_df = sample_df.tail(min(200, len(sample_df)))

                X_sample = sample_df[get_feature_columns(sample_df)]
                explainer = shap.Explainer(trained_model, X_sample)
                shap_values = explainer(X_sample)

                shap_df = pd.DataFrame(
                    np.abs(shap_values.values).mean(axis=0),
                    index=X_sample.columns,
                    columns=["mean_abs_shap"]
                ).sort_values("mean_abs_shap", ascending=False).reset_index()
                shap_df.columns = ["feature", "mean_abs_shap"]

                fig = px.bar(
                    shap_df.head(15),
                    x="feature",
                    y="mean_abs_shap",
                    title="SHAP 重要性 Top 15"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "下載 SHAP 重要性",
                    to_csv_download(shap_df),
                    file_name="shap_importance.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.warning(f"SHAP 分析失敗：{e}")
    else:
        st.caption("若已安裝 shap，可顯示更完整的模型解釋。")

st.markdown("---")
st.caption("ESG-AI Smart Portfolio Dashboard | 短樣本可跑版（建議搭配本機 CSV）")