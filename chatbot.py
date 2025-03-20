import streamlit as st
import openai
from openai import OpenAI
import yfinance as yf
import json
import time
import os
from dotenv import load_dotenv


# Function to determine the API key source
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, AttributeError, FileNotFoundError):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY not found in .env file or environment variables!")
            st.stop()
        return api_key


# Initialize OpenAI client
client = OpenAI(api_key=get_api_key())


# Function to get stock price by ticker
def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        price = stock.history(period="1d")["Close"].iloc[-1]
        return round(price, 2)
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"


# Function specification for OpenAI Responses API
stock_price_function = {
    "type": "function",
    "name": "get_stock_price",
    "description": "Get the most recent closing price of a stock by its ticker symbol using Yahoo Finance data",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol (e.g., AAPL for Apple)",
            }
        },
        "required": ["ticker"],
        "additionalProperties": False,
    },
}


# Check if query is related to stocks/crypto/trading
def is_related_to_stocks_crypto(query):
    # Existing keywords
    keywords = [
        "stock",
        "stocks",
        "crypto",
        "cryptocurrency",
        "trade",
        "trading",
        "market",
        "price",
        "invest",
        "investment",
        "bitcoin",
        "ethereum",
        "portfolio",
        "bull",
        "bear",
        "exchange",
        "gold",
        "XAUUSD",
    ]

    # Add company/business-related keywords
    company_keywords = [
        "company",
        "business",
        "corporation",
        "inc",
        "ltd",
        "information",
        "operations",
        "industry",
        "revenue",
        "products",
        "services",
    ]

    query_lower = query.lower()

    # Check for direct financial keywords
    if any(keyword in query_lower for keyword in keywords):
        return True

    # Check for company-related keywords
    if any(keyword in query_lower for keyword in company_keywords):
        # Extract potential company names (simple heuristic: capitalized words)
        words = query.split()
        potential_companies = [
            word for word in words if word[0].isupper() and len(word) > 2
        ]

        # Check if any potential company name has a valid stock ticker
        for company in potential_companies:
            try:
                ticker = yf.Ticker(company.upper())  # Try as a ticker symbol
                if ticker.info and "symbol" in ticker.info:  # Valid ticker
                    return True
            except Exception:
                # If not a ticker, check a predefined list or skip
                pass

    # Add a small list of well-known companies with stocks (optional fallback)
    known_companies = [
        "tesla",
        "apple",
        "microsoft",
        "google",
        "amazon",
        "facebook",
        "nvidia",
        "coinbase",
        "binance",
        "netflix",
        "ford",
        "gm",
        "boeing",
        "hp",
    ]

    if any(company in query_lower for company in known_companies):
        return True

    return False


# Streaming response generator with function calling
def response_generator(query):
    if not is_related_to_stocks_crypto(query):
        for (
            word
        ) in "I can only answer questions about stocks, cryptocurrency, or trading. Please ask about one of those topics!".split():
            yield word + " "
            time.sleep(0.1)
        return

    # Initial call to LLM
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. Use the get_stock_price function when asked for a stock price. Provide a clear comparison when asked about multiple stocks.\n\nUser: {query}",
        tools=[stock_price_function],
        stream=False,
    )
    print("First Response: " + str(response.output))
    print("-" * 60)

    # Check if response contains output
    if not response.output or len(response.output) == 0:
        for word in "No response received from the API".split():
            yield word + " "
            time.sleep(0.1)
        return

    # Function to process text with Markdown-like formatting
    # Function to process text with Markdown-like formatting
    def process_text(text):
        # Split text into lines for processing
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            if not line.strip():
                # Preserve empty lines as breaks
                processed_lines.append("<br>")
                continue

            # Handle headings
            if line.startswith("### "):
                line = f"<h3>{line[4:].strip()}</h3>"
            elif line.startswith("## "):
                line = f"<h2>{line[3:].strip()}</h2>"
            elif line.startswith("# "):
                line = f"<h1>{line[2:].strip()}</h1>"
            else:
                # Wrap non-heading lines in a paragraph tag for consistency
                line = f"<p>{line.strip()}</p>"

            # Process inline Markdown within the line
            # Bold (**text** or __text__)
            while "**" in line:
                line = line.replace("**", "<b>", 1).replace("**", "</b>", 1)
            while "__" in line:
                line = line.replace("__", "<b>", 1).replace("__", "</b>", 1)

            # Italics (*text* or _text_)
            while "*" in line and line.count("*") >= 2:
                line = line.replace("*", "<i>", 1).replace("*", "</i>", 1)
            while (
                "_" in line and line.count("_") >= 2 and "__" not in line
            ):  # Avoid conflict with bold
                line = line.replace("_", "<i>", 1).replace("_", "</i>", 1)

            processed_lines.append(line)

        # Join lines with breaks where needed
        return "".join(processed_lines)

    # Handle multiple tool calls
    tool_calls = [
        output
        for output in response.output
        if hasattr(output, "type") and output.type == "function_call"
    ]

    if tool_calls:
        # Process all stock prices
        stock_prices = {}
        for tool_call in tool_calls:
            if tool_call.name == "get_stock_price":
                args = json.loads(tool_call.arguments)
                ticker = args["ticker"]
                price = get_stock_price(ticker)
                stock_prices[ticker] = price

        # Prepare tool results for follow-up
        tool_results = "\n".join(
            [
                f"The latest price for {ticker} is ${price}"
                for ticker, price in stock_prices.items()
            ]
        )

        # Follow-up call with explicit instruction to summarize results
        follow_up_response = client.responses.create(
            model="gpt-4o-mini",
            input=f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading. The user asked: '{query}'. Using the tool results below, provide a concise text response summarizing the information. Do not invoke additional tool calls unless explicitly requested.\n\nTool results:\n{tool_results}\n\nNow, respond to the user with the stock prices in a clear, natural language format.",
            tools=[
                stock_price_function
            ],  # Still provide tools in case they're needed later
            stream=False,
        )
        print("Follow-up Response: " + str(follow_up_response))
        print("-" * 60)
        print(
            f"System: You are a financial assistant specializing in stocks, cryptocurrency, and trading.\n\nUser: {query}\n\nTool results:\n{tool_results}"
        )
        print("-" * 60)

        # Handle follow-up response
        if (
            follow_up_response.output
            and len(follow_up_response.output) > 0
            and hasattr(follow_up_response.output[0], "content")
            and len(follow_up_response.output[0].content) > 0
        ):
            raw_text = follow_up_response.output[0].content[0].text
            formatted_text = process_text(raw_text)
            words = formatted_text.split(" ")
            for word in words:
                yield word + " "
                time.sleep(0.1)
        else:
            # Fallback: If no text response, yield the tool results directly
            formatted_text = process_text(tool_results)
            words = formatted_text.split(" ")
            for word in words:
                yield word + " "
                time.sleep(0.1)
    # Check if it's a direct response (ResponseOutputMessage)
    elif (
        hasattr(response.output[0], "content")
        and len(response.output[0].content) > 0
        and hasattr(response.output[0].content[0], "text")
    ):
        raw_text = response.output[0].content[0].text
        formatted_text = process_text(raw_text)
        words = formatted_text.split(" ")
        for word in words:
            yield word + " "
            time.sleep(0.1)
    else:
        for word in "Error: Unable to process the response".split():
            yield word + " "
            time.sleep(0.1)


# Chatbot state initialization
def init_chatbot_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


# Load chat CSS
def load_chat_css():
    chat_css = """
    <style>
    .chat-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }
    .chat-row {
        display: flex;
        margin: 5px;
        width: 100%;
        align-items: flex-end;
    }
    .row-reverse {
        flex-direction: row-reverse;
    }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        border: 1px solid transparent;
        padding: 8px 15px;
        margin: 5px 7px;
        max-width: 70%;
        border-radius: 20px;
    }
    .assistant-bubble {
        background-color: #eee;
        margin-right: 25%;
        position: relative;
    }
    .assistant-bubble:before {
        content: "";
        position: absolute;
        z-index: 0;
        bottom: 0;
        left: -7px;
        height: 20px;
        width: 20px;
        background: #eee;
        border-bottom-right-radius: 15px;
    }
    .assistant-bubble:after {
        content: "";
        position: absolute;
        z-index: 1;
        bottom: 0;
        left: -10px;
        width: 10px;
        height: 20px;
        background: white;
        border-bottom-right-radius: 10px;
    }
    .user-bubble {
        color: white;
        background-color: #1F8AFF;
        margin-left: 25%;
    }
    .user-bubble:before {
        content: "";
        position: absolute;
        z-index: 0;
        bottom: -10px;
        right: 20px;
        height: 20px;
        width: 20px;
        background: #1F8AFF;
        border-bottom-left-radius: 15px;
        align-items: flex-end;
    }
    .user-bubble:after {
        content: "";
        position: absolute;
        z-index: 1;
        bottom: -10px;
        right: 20px;
        width: 10px;
        height: 20px;
        background: white;
        border-bottom-left-radius: 10px;
        align-items: flex-end;
    }
    .chat-icon {
        width: 28px !important;
        height: 28px !important;
        padding: 5px;
        margin-top: 5px !important;
        flex-shrink: 0;
    }
    .user-icon {
        color: rgb(31, 138, 255) !important;
    }
    .assistant-icon {
        color: rgb(64, 64, 64);
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)


# Chat icon SVGs
def get_chat_icon(role):
    if role == "user":
        return """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M12 4a3.5 3.5 0 1 0 0 7a3.5 3.5 0 0 0 0-7M6.5 7.5a5.5 5.5 0 1 1 11 0a5.5 5.5 0 0 1-11 0M3 19a5 5 0 0 1 5-5h8a5 5 0 0 1 5 5v3H3zm5-3a3 3 0 0 0-3 3v1h14v-1a3 3 0 0 0-3-3z"/></svg>"""
    return """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M18.5 10.255q0 .067-.003.133A1.54 1.54 0 0 0 17.473 10q-.243 0-.473.074V5.75a.75.75 0 0 0-.75-.75h-8.5a.75.75 0 0 0-.75.75v4.505c0 .414.336.75.75.75h8.276l-.01.025l-.003.012l-.45 1.384l-.01.026l-.019.053H7.75a2.25 2.25 0 0 1-2.25-2.25V5.75A2.25 2.25 0 0 1 7.75 3.5h3.5v-.75a.75.75 0 0 1 .649-.743L12 2a.75.75 0 0 1 .743.649l.007.101l-.001.75h3.5a2.25 2.25 0 0 1 2.25 2.25zm-5.457 3.781l.112-.036H6.254a2.25 2.25 0 0 0-2.25 2.25v.907a3.75 3.75 0 0 0 1.305 2.844c1.563 1.343 3.802 2 6.691 2c2.076 0 3.817-.339 5.213-1.028a1.55 1.55 0 0 1-1.169-1.003l-.004-.012l-.03-.093c-1.086.422-2.42.636-4.01.636c-2.559 0-4.455-.556-5.713-1.638a2.25 2.25 0 0 1-.783-1.706v-.907a.75.75 0 0 1 .75-.75H12v-.003a1.54 1.54 0 0 1 1.031-1.456zM10.999 7.75a1.25 1.25 0 0 0-2.499 0a1.25 1.25 0 0 0 2.499 0m3.243-1.25a1.25 1.25 0 1 1 0 2.499a1.25 1.25 0 0 1 0-2.499m1.847 10.912a2.83 2.83 0 0 0-1.348-.955l-1.377-.448a.544.544 0 0 1 0-1.025l1.377-.448a2.84 2.84 0 0 0 1.76-1.762l.01-.034l.449-1.377a.544.544 0 0 1 1.026 0l.448 1.377a2.84 2.84 0 0 0 1.798 1.796l1.378.448l.027.007a.544.544 0 0 1 0 1.025l-1.378.448a2.84 2.84 0 0 0-1.798 1.796l-.447 1.377a.55.55 0 0 1-.2.263a.544.544 0 0 1-.827-.263l-.448-1.377a2.8 2.8 0 0 0-.45-.848m7.694 3.801l-.765-.248a1.58 1.58 0 0 1-.999-.998l-.249-.765a.302.302 0 0 0-.57 0l-.249.764a1.58 1.58 0 0 1-.983.999l-.766.248a.302.302 0 0 0 0 .57l.766.249a1.58 1.58 0 0 1 .999 1.002l.248.764a.303.303 0 0 0 .57 0l.25-.764a1.58 1.58 0 0 1 .998-.999l.766-.248a.302.302 0 0 0 0-.57z"/></svg>"""


# Create message div
def create_message_div(role, content):
    icon = get_chat_icon(role)
    chat_icon_class = f"chat-icon {'user-icon' if role == 'user' else 'assistant-icon'}"
    return f"""
    <div class="chat-row {'row-reverse' if role == 'user' else ''}">
        <div class="chat-icon-container">
            <div class="{chat_icon_class}">{icon}</div>
        </div>
        <div class="chat-bubble {'user-bubble' if role == 'user' else 'assistant-bubble'}">
            {content}
        </div>
    </div>
    """


# Main chatbot UI
def show_chatbot_ui():
    init_chatbot_state()
    load_chat_css()

    st.title("💬 Stock/Crypto Chatbot")

    # Center container
    with st.container():
        # Chat messages
        messages_container = st.container(height=400)
        with messages_container:
            for msg in st.session_state.messages:
                st.markdown(
                    create_message_div(msg["role"], msg["content"]),
                    unsafe_allow_html=True,
                )

        # Input at bottom
        if user_query := st.chat_input("Ask about stocks, crypto, or trading:"):
            st.session_state.messages.append({"role": "user", "content": user_query})

            with messages_container:
                for msg in st.session_state.messages:
                    st.markdown(
                        create_message_div(msg["role"], msg["content"]),
                        unsafe_allow_html=True,
                    )
                streaming_placeholder = st.empty()

            with messages_container:
                with st.spinner("Thinking..."):
                    response_gen = response_generator(user_query)
                    full_response = ""
                    try:
                        first_chunk = next(response_gen)
                        full_response = first_chunk
                    except StopIteration:
                        first_chunk = ""
                        full_response = ""
                    streaming_div = create_message_div("assistant", full_response)
                    streaming_placeholder.markdown(
                        streaming_div, unsafe_allow_html=True
                    )

            for chunk in response_gen:
                full_response += chunk
                streaming_div = create_message_div("assistant", full_response)
                streaming_placeholder.markdown(streaming_div, unsafe_allow_html=True)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            st.rerun()


# Streamlit app
# st.set_page_config(layout="centered")
show_chatbot_ui()


# WORKING with multiple stocks and markdown formatting
