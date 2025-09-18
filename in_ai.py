import os
import yfinance as yf
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq

# ==============================
# Config & Setup
# ==============================
st.set_page_config(page_title="AI Investment Dashboard", page_icon="📊", layout="wide")
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # Only from env, no UI input

if not groq_api_key:
    st.error("❌ Groq API Key not found. Please set GROQ_API_KEY in your .env or environment variables.")
    st.stop()

# ==============================
# Helper functions
# ==============================
def get_company_info(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "Company": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market Cap": f"${info.get('marketCap'):,}" if info.get("marketCap") else "N/A",
    }

def get_fundamentals(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "Previous Close": info.get("previousClose"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price/Book": info.get("priceToBook"),
    }

def get_recent_prices(symbol, period="6mo", interval="1mo"):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    hist = hist.tail(6).reset_index()[["Date", "Close", "Volume"]]
    hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
    return hist

# ==============================
# UI Layout
# ==============================
st.title("📊 AI Investment Dashboard")
st.caption("Compare company fundamentals, stock performance, and AI analysis.")

with st.sidebar:
    st.header("📈 Stock Panel")
    stock1 = st.text_input("Stock 1 Symbol", value="AAPL")
    stock2 = st.text_input("Stock 2 Symbol", value="MSFT")
    run_analysis = st.button("Run Analysis")

# ==============================
# Main Content
# ==============================
if run_analysis and stock1 and stock2:
    col1, col2 = st.columns(2)

    # --- Company Info ---
    with col1:
        st.subheader(f"🏢 {stock1} Company Info")
        info1 = get_company_info(stock1)
        st.table(pd.DataFrame(info1.items(), columns=["Metric", "Value"]))

    with col2:
        st.subheader(f"🏢 {stock2} Company Info")
        info2 = get_company_info(stock2)
        st.table(pd.DataFrame(info2.items(), columns=["Metric", "Value"]))

    # --- Fundamentals ---
    st.markdown("## 📌 Fundamentals")
    fcol1, fcol2 = st.columns(2)

    with fcol1:
        fund1 = get_fundamentals(stock1)
        st.metric("Previous Close", fund1["Previous Close"])
        st.metric("Trailing P/E", fund1["Trailing P/E"])
        st.metric("Forward P/E", fund1["Forward P/E"])
        st.metric("Price/Book", fund1["Price/Book"])

    with fcol2:
        fund2 = get_fundamentals(stock2)
        st.metric("Previous Close", fund2["Previous Close"])
        st.metric("Trailing P/E", fund2["Trailing P/E"])
        st.metric("Forward P/E", fund2["Forward P/E"])
        st.metric("Price/Book", fund2["Price/Book"])

    # --- Recent Prices ---
    st.markdown("## 📈 Recent Price Trend")
    prices1 = get_recent_prices(stock1)
    prices2 = get_recent_prices(stock2)

    st.line_chart(prices1.set_index("Date")["Close"], height=300)
    st.line_chart(prices2.set_index("Date")["Close"], height=300)

    # --- AI Analysis ---
    st.markdown("## 🤖 AI Investment Analysis")

    assistant = Agent(model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key))

    query = (
        f"Compare {stock1} ({info1['Company']}) and {stock2} ({info2['Company']}). "
        "Analyze fundamentals, recent price performance, risks, and give an investment recommendation. "
        "Be concise and format your response as a professional financial analyst report with bullet points and sections."
    )

    with st.spinner("AI is analyzing the stocks..."):
        response = assistant.run(query, stream=False)

    st.markdown(response.content)
