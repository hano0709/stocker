import os
import sys
import re
import google.generativeai as genai
import gradio as gr

# Ensure local module import works
sys.path.append(os.getcwd())
from stock_prediction_transformer import StockManager

# ✅ Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Uses environment variable

# ✅ Load Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

# ✅ Initialize Stock System
stock_manager = StockManager()

# 🔧 Ensure required folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ✅ Extract ticker from Gemini response
def extract_ticker_from_text(text):
    text = text.replace("**", "").replace(".", "").strip()
    match = re.search(r'\b[A-Z]{1,5}\b', text)
    if match:
        return match.group(0)
    return text.strip()

# ✅ Ask Gemini for ticker
def get_ticker(stock_name):
    try:
        prompt = f"What is the stock ticker symbol of {stock_name}?"
        response = model.generate_content(prompt)
        return extract_ticker_from_text(response.text.strip())
    except Exception as e:
        return None

# ✅ Handle prediction
def handle_prediction(stock_name, duration):
    ticker = get_ticker(stock_name)
    if not ticker:
        return f"❌ Couldn't determine ticker for '{stock_name}'."
    
    _, predictions, info = stock_manager.predict_stock(ticker, duration)
    if predictions is None:
        return "❌ Prediction failed."
    
    predicted_price = predictions[-1][0]
    current_price = info.get("current_price", 0)
    accuracy = info.get("accuracy", 0) * 100

    change_str = f"({(predicted_price - current_price) / current_price * 100:+.2f}%) change from current price ${current_price:.2f}" if current_price > 0 else "(Current price unavailable)"

    return f"✅ Predicted price of {ticker} after {duration}: **${predicted_price:.2f}**\n{change_str}\nModel Accuracy: {accuracy:.2f}%"

# ✅ Gemini chatbot logic with conversational flow
session_state = {"step": 0, "stock_name": None}

def chatbot(user_message, history):
    try:
        user_message_lower = user_message.lower()

        # 🔹 Step 1: Ask for stock name
        if session_state["step"] == 0:
            session_state["step"] = 1
            return "Which stock are you interested in? (e.g., Apple, Google, Nvidia)"

        # 🔹 Step 2: Store stock name & ask for duration
        elif session_state["step"] == 1:
            session_state["stock_name"] = user_message.strip()
            session_state["step"] = 2
            return "For how long would you like a prediction? (e.g., 1d, 1w, 1mo)"

        # 🔹 Step 3: Convert to ticker & predict
        elif session_state["step"] == 2:
            stock_name = session_state["stock_name"]
            duration = user_message.strip()
            session_state["step"] = 0  # Reset for next query
            
            ticker = get_ticker(stock_name)
            if not ticker:
                return f"❌ Couldn't determine the ticker for '{stock_name}'. Try again."
            
            prediction = handle_prediction(stock_name, duration)
            return f"📈 Prediction for {stock_name} ({ticker}) in {duration}:\n\n{prediction}"

        # Default fallback
        return "I'm not sure what you mean. Let's start over! Which stock are you interested in?"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ✅ Gradio Chat Interface
chat_app = gr.ChatInterface(fn=chatbot, title="Gemini Stock Chatbot", description="Chat with Gemini AI about stock predictions!")

if __name__ == "__main__":
    chat_app.launch(server_name="0.0.0.0", server_port=7860)
