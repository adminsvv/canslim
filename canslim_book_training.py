

import pandas as pd
from datetime import datetime, timedelta,date,time as dt_time
from datetime import datetime, time
import numpy as np
import os
pd.options.display.float_format = '{:.2f}'.format
import time

from pandas.io.formats.style import Styler

from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Suppress all warnings




from openai import OpenAI
import openai


import openai

import os
import pickle


api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)
print("key generrated)")

CHAPTERS_TO_PROCESS = {
    1: {"start": 14, "end": 64, "title": "The Greatest Stock-Picking Secrets"},
    2: {"start": 65, "end": 133, "title": "How to Read Charts Like a Pro and Improve Your Selection and Timing"},
    3: {"start": 134, "end": 142, "title": "C = Current Big or Accelerating Quarterly Earnings and Sales"},
    4: {"start": 143, "end": 153, "title": "A = Annual Earnings Increases: Look for Big Growth"},
    5: {"start": 154, "end": 166, "title": "N = New Companies, New Products, New Management, New Highs"},
    6: {"start": 167, "end": 172, "title": "S = Supply and Demand: Big Volume Demand at Key Points"},
    7: {"start": 173, "end": 180, "title": "L = Leader or Laggard: Which Is Your Stock?"},
    8: {"start": 181, "end": 187, "title": "I = Institutional Sponsorship"},
    9: {"start": 188, "end": 232, "title": "M = Market Direction: How You Can Determine It"},
    10: {"start": 233, "end": 248, "title": "When You Must Sell and Cut Every Loss … Without Exception"},
    11: {"start": 249, "end": 275, "title": "When to Sell and Take Your Worthwhile Profits"},
    12: {"start": 276, "end": 307, "title": "Money Management: Diversification, Margin, Options, IPOs, Foreign Stocks"},
    13: {"start": 308, "end": 314, "title": "Twenty-One Costly Common Mistakes Most Investors Make"},
    14: {"start": 315, "end": 368, "title": "More Models of Great Stock Market Winners"},
    15: {"start": 369, "end": 390, "title": "Picking the Best Market Themes, Sectors, and Industry Groups"},
}

with open("chapter_embeddings.pkl", "rb") as f:
    chapter_embeddings = pickle.load(f)

CREDENTIALS = {
    "canslim": "canslim_gpt"

}

def login_block():
    """Returns True when user is authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    # ── Login form ──
    st.title("🔐 Login required")
    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        submit = st.form_submit_button("Log in")

    if submit:
        if user in CREDENTIALS and pwd == CREDENTIALS[user]:
            st.session_state["authenticated"] = True
            #st.experimental_rerun()
        else:
            st.error("❌ Incorrect username or password")
    return False


# Call the blocker right at the top of your script
if not login_block():
    st.stop()

# PDF_PATH = "D:/Chat Gpt/CANSLIM/How-to-Make-Money-in-Stocks-PDF-Book.pdf"
# CHAPTERS_TO_PROCESS = {
#     1: {"start": 14, "end": 64, "title": "The Greatest Stock-Picking Secrets"},
#     2: {"start": 65, "end": 133, "title": "How to Read Charts Like a Pro"},
#     3: {"start": 134, "end": 142, "title": "C = Current Big or Accelerating Quarterly Earnings and Sales"},
#     4: {"start": 143, "end": 153, "title": "A = Annual Earnings Increases: Look for Big Growth"},
#     5: {"start": 154, "end": 166, "title": "N = New Companies, New Products, New Management"},
# }

# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
# EMBED_MODEL = "text-embedding-3-small"

# # ─── Open book ───────────────────────────────
# doc = fitz.open(PDF_PATH)
# enc = tiktoken.encoding_for_model(EMBED_MODEL)

# # ─── Helper: Chunker ──────────────────────────
# def chunk_tokens(tokens, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     step = size - overlap
#     for i in range(0, len(tokens), step):
#         yield tokens[i: i + size]

# # ─── Storage ──────────────────────────────────
# chapter_embeddings = {}

# # ─── Process Each Chapter ─────────────────────
# for chapter_no, meta in CHAPTERS_TO_PROCESS.items():
#     full_text = ""
#     for pg in range(meta["start"], meta["end"] + 1):
#         full_text += doc.load_page(pg).get_text("text") + "\n"
#     full_text = full_text.strip()
    
#     tokens = enc.encode(full_text)
#     chunks = [enc.decode(window) for window in chunk_tokens(tokens)]
    
#     # Embed chunks
#     embedded_chunks = []
#     batch_size = 50
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i+batch_size]
#         response = client.embeddings.create(
#         input=batch,
#         model="text-embedding-3-small"
#             )
#         embedded_chunks.extend(response.data)
    
#     # Save embeddings
#     chapter_embeddings[chapter_no] = {
#         "title": meta["title"],
#         "chunks": chunks,
#         "embeddings": [e.embedding for e in embedded_chunks]
#     }
    
#     print(f"✅ Chapter {chapter_no}: {meta['title']} processed with {len(chunks)} chunks.")


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def match_user_question_in_chapter(user_question, chapter_no, top_k=6):
    """
    Match user question inside a specific chapter.
    """
    query_emb = get_embedding(user_question)
    
    similarities = []
    for idx, emb in enumerate(chapter_embeddings[chapter_no]["embeddings"]):
        score = cosine_similarity([query_emb], [emb])[0][0]
        similarities.append((idx, score))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:top_k]
    
    result = []
    for idx, score in top_matches:
        result.append({
            "chunk_id": idx,
            "similarity": round(score, 4),
            "text": chapter_embeddings[chapter_no]["chunks"][idx]
        })
        
    return result

tools = [
    {
        "type": "function",
        "name": "search_greatest_stock_picking_secrets",
        "description": "Search information from Chapter 1: The Greatest Stock-Picking Secrets.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_chart_reading_and_timing",
        "description": "Search information from Chapter 2: How to Read Charts Like a Pro and Improve Your Selection and Timing.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_current_earnings_growth",
        "description": "Search information from Chapter 3: C = Current Big or Accelerating Quarterly Earnings and Sales.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_annual_earnings_growth",
        "description": "Search information from Chapter 4: A = Annual Earnings Increases: Look for Big Growth.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_new_companies_and_products",
        "description": "Search information from Chapter 5: N = New Companies, New Products, New Management, New Highs.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_supply_and_demand",
        "description": "Search information from Chapter 6: S = Supply and Demand: Big Volume Demand at Key Points.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_leader_or_laggard",
        "description": "Search information from Chapter 7: L = Leader or Laggard: Which Is Your Stock?.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_institutional_sponsorship",
        "description": "Search information from Chapter 8: I = Institutional Sponsorship.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_market_direction",
        "description": "Search information from Chapter 9: M = Market Direction: How You Can Determine It.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_cutting_losses",
        "description": "Search information from Chapter 10: When You Must Sell and Cut Every Loss.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_taking_profits",
        "description": "Search information from Chapter 11: When to Sell and Take Your Worthwhile Profits.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_money_management",
        "description": "Search information from Chapter 12: Money Management: Diversification, Margin, Options, IPOs, Foreign Stocks.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_common_investor_mistakes",
        "description": "Search information from Chapter 13: Twenty-One Costly Common Mistakes Most Investors Make.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_models_of_stock_winners",
        "description": "Search information from Chapter 14: More Models of Great Stock Market Winners.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    },
    {
        "type": "function",
        "name": "search_best_sectors_and_themes",
        "description": "Search information from Chapter 15: Picking the Best Market Themes, Sectors, and Industry Groups.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    }
]



def search_greatest_stock_picking_secrets(): return 1
def search_chart_reading_and_timing(): return 2
def search_current_earnings_growth(): return 3
def search_annual_earnings_growth(): return 4
def search_new_companies_and_products(): return 5
def search_supply_and_demand(): return 6
def search_leader_or_laggard(): return 7
def search_institutional_sponsorship(): return 8
def search_market_direction(): return 9
def search_cutting_losses(): return 10
def search_taking_profits(): return 11
def search_money_management(): return 12
def search_common_investor_mistakes(): return 13
def search_models_of_stock_winners(): return 14
def search_best_sectors_and_themes(): return 15

function_map = {
    "search_greatest_stock_picking_secrets": search_greatest_stock_picking_secrets,
    "search_chart_reading_and_timing": search_chart_reading_and_timing,
    "search_current_earnings_growth": search_current_earnings_growth,
    "search_annual_earnings_growth": search_annual_earnings_growth,
    "search_new_companies_and_products": search_new_companies_and_products,
    "search_supply_and_demand": search_supply_and_demand,
    "search_leader_or_laggard": search_leader_or_laggard,
    "search_institutional_sponsorship": search_institutional_sponsorship,
    "search_market_direction": search_market_direction,
    "search_cutting_losses": search_cutting_losses,
    "search_taking_profits": search_taking_profits,
    "search_money_management": search_money_management,
    "search_common_investor_mistakes": search_common_investor_mistakes,
    "search_models_of_stock_winners": search_models_of_stock_winners,
    "search_best_sectors_and_themes": search_best_sectors_and_themes,
}
def answer_user_question(user_question: str,model_choice):
    input_messages = [{"role": "user", "content": user_question}]

    response = client.responses.create(
        model="gpt-4.1",
        input=input_messages,
        tools=tools,
        tool_choice="required"
    
    )

    user_question=input_messages[0]['content']
    #print(user_question)
    #print(response.output)
    for call in response.output:
        st.write(f"🧩 Function Name: {call.name}")
        print(f"📦 Arguments: {call.arguments}\n")


    executed_chapters = []

    for call in response.output:
        func = function_map.get(call.name)
        if func:
            chapter_no = func()  # execute the function
            executed_chapters.append(chapter_no)
        else:
            print(f"⚠️ Unknown function: {call.name}")

    if not executed_chapters:
        raise Exception("No chapter selected!")

    print(f"✅ Chapters selected: {executed_chapters}")

    all_matches = []

    for chapter_no in executed_chapters:
        chapter_matches = match_user_question_in_chapter(user_question, chapter_no=chapter_no, top_k=6)  # top_k per chapter
        all_matches.extend(chapter_matches)

    # --- Step 3: Sort matches by similarity ---
    all_matches_sorted = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)

    def build_context_from_matches(matches):
        context_parts = []
        for idx, match in enumerate(matches, start=1):
            part = f"""### Chunk {match['chunk_id']} (Score: {match['similarity']})
    {match['text'].strip()}
    """
            context_parts.append(part)
        return "\n\n---\n\n".join(context_parts)

    final_context = build_context_from_matches(all_matches_sorted)

    # (ready to pass into GPT)
    print(final_context)

    print(user_question)

    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {"role": "system", "content": "You are a trading mentor answering user questions only based on the provided book excerpts. If the context is not enough, say: 'Based on the given information, I cannot answer this.' Do not invent facts."},
            {"role": "user", "content": user_question},
            {"role": "system", "content": f"Below are selected excerpts from Chapter 3:\n\n{final_context}"}
        ],
        temperature=0.2
    )


    #final_answer=response.choices[0].message.content.strip()
    return response

st.set_page_config(page_title="CANSLIM Trading Mentor", page_icon="📚")

st.title("📚 CANSLIM Trading Mentor")
st.subheader("Ask anything based on 'How to Make Money in Stocks' 📈")

models = {
    "gpt-4.1":       {"Input": 2.00,  "Cached": 0.50,  "Output": 8.00},
    "gpt-4.1-mini":  {"Input": 0.40,  "Cached": 0.10,  "Output": 1.60},
    "gpt-4.1-nano":  {"Input": 0.10,  "Cached": 0.025, "Output": 0.40},
}

# dropdown with default = gpt-4.1-mini
model_choice = st.selectbox(
    "Model:",
    list(models.keys()),
    index=list(models.keys()).index("gpt-4.1-mini")
)

price = models[model_choice]

st.write("Model Cost")
# ─── show one-row price bar ──────────────────
c1, c2 = st.columns(2)
c1.metric("Input\n($ / 1M tok)",   f"${price['Input']:.1f}")
c2.metric("Output",                f"${price['Output']:.1f}")

first_half = {k: v for k, v in list(CHAPTERS_TO_PROCESS.items())[:7]}
second_half = {k: v for k, v in list(CHAPTERS_TO_PROCESS.items())[7:]}
col1, col2 = st.columns(2)
# Display first half in left column
with col1:
    for chapter_no, meta in first_half.items():
        st.markdown(f"**{chapter_no}. {meta['title']}**")

# Display second half in right column
with col2:
    for chapter_no, meta in second_half.items():
        st.markdown(f"**{chapter_no}. {meta['title']}**")

st.markdown("---")  # separator line

user_question = st.text_area("💬 Enter your question", height=150)

if st.button("Get Answer 🚀"):
    if user_question.strip() == "":
        st.error("Please enter a question first!")
    else:
        with st.spinner('🔎 Thinking and retrieving best answer...'):
            final_answer = answer_user_question(user_question,model_choice)
        st.success("✅ Here’s your answer!")
        st.markdown(final_answer.choices[0].message.content.strip())
        prompt_tok     = final_answer.usage.prompt_tokens
        completion_tok = final_answer.usage.completion_tokens

        rate_in   = models[model_choice]["Input"]   # $ per 1 M input tokens
        rate_out  = models[model_choice]["Output"]  # $ per 1 M output tokens

        cost_in   = (prompt_tok     / 1_000_000) * rate_in *90
        cost_out  = (completion_tok / 1_000_000) * rate_out * 90
        cost_tot  = cost_in + cost_out

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Input tokens",     f"{prompt_tok:,}")
        colB.metric("Output tokens",    f"{completion_tok:,}")
        colC.metric("Total cost (INR)",       f"{cost_tot:,.4f}")
        colD.metric("Model",            model_choice)


