# app.py
import os
import time
import traceback
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pymongo import MongoClient
from bson.objectid import ObjectId

# langchain imports
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration (via env variables) ---
MONGODB_URI = os.environ.get("MONGODB_URI")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index")

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "https://apnabzaar.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy singletons to reduce memory overhead ---


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI environment variable not set")
    return MongoClient(MONGODB_URI)


@lru_cache(maxsize=1)
def get_embeddings():
    # small model, cached once
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = get_embeddings()
    try:
        vs = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        return vs
    except Exception as e:
        # surface clear message but do not crash import-time (we raise so endpoints can handle)
        raise RuntimeError(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}")


@lru_cache(maxsize=1)
def get_llm():
    # cached LLM client
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# bounded cache for searches
@lru_cache(maxsize=128)
def cached_search(query: str):
    vs = get_vector_store()
    return vs.similarity_search(query, k=5)


# Prompt template
PROMPT = PromptTemplate(
    template="""
You are a helpful chatbot for an e-commerce website. Use ONLY the information found in the provided context. Answer concisely in 1–2 lines.

If the context does not contain enough information to answer, reply exactly: "No data found".

Rules:
1. If the user requests a product recommendation, recommend up to 3 products that best match the user's preferences and needs, using only product fields present in the context along with the product data and link to the product.
   Product link format:
   https://apnabzaar.netlify.app/productdetail/product_id
   Replace `product_id` with the product's id from the Product Data.
2. If the user asks for a product that is not available in the context, reply exactly:
"We are sorry, the product you requested is currently not available on our site. However, we value your interest and would be happy to assist you with similar products or alternatives that meet your needs. Please let us know what you're looking for, and we'll do our best to help you find a suitable option."
3. If the user asks for order details, return only order information present in the order data. Do NOT invent missing fields.
4. If the user asks for a product's price, reply with the price, Name of product and the product link in this exact format:
https://apnabzaar.netlify.app/productdetail/product_id
5. Do not provide any information that is not present in the context. Keep it to 1–2 lines.

Context:
{context}

Question: {question}

Product Data : {products}
Order Data : {orders}
""",
    input_variables=["context", "question", "products", "orders"],
)


# --- DB snippet fetchers (tiny payloads) ---
def fetch_product_snippet(limit: int = 10) -> str:
    client = get_mongo_client()
    db = client["ECommerce"]
    col = db["products"]
    cursor = col.find(
        {"isActive": True},
        {"name": 1, "price": 1, "category": 1, "description": 1},
    ).limit(limit)
    parts = []
    for p in cursor:
        pid = str(p.get("_id"))
        parts.append(
            f"ProductID: {pid} | Name: {p.get('name','')} | Price: {p.get('price','')} | Category: {p.get('category','')} | Description: {p.get('description','')}"
        )
    return "\n".join(parts)


def fetch_user_orders_snippet(user_id: str, limit: int = 5) -> str:
    client = get_mongo_client()
    db = client["ECommerce"]
    col = db["users"]

    try:
        uid = ObjectId(user_id)
    except Exception:
        uid = user_id

    user = col.find_one({"_id": uid}, {"orders": 1})
    if not user or "orders" not in user:
        return ""
    parts = []
    for o in user.get("orders", [])[:limit]:
        parts.append(str(o))
    return "\n".join(parts)


# --- Robust chat (tries multiple invocation methods, logs tracebacks) ---
async def chat_ai_async(user_id: str, question: str):
    start_ts = time.time()
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    try:
        # similarity search (safe)
        try:
            docs = cached_search(question)
        except Exception as e:
            print("DEBUG: vector search failed:", repr(e))
            docs = []

        context = "\n".join([d.page_content for d in docs]) if docs else ""
        products_snip = fetch_product_snippet(limit=10)
        orders_snip = fetch_user_orders_snippet(user_id, limit=5)

        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=PROMPT)

        result = None
        exc = None
        try:
            if hasattr(chain, "ainvoke"):
                result = await chain.ainvoke(
                    {"context": context, "question": question, "products": products_snip, "orders": orders_snip}
                )
            elif hasattr(chain, "invoke"):
                result = chain.invoke(
                    {"context": context, "question": question, "products": products_snip, "orders": orders_snip}
                )
            else:
                # fallback: try direct llm predict if chain isn't available
                prompt_text = PROMPT.format(context=context, question=question, products=products_snip, orders=orders_snip)
                if hasattr(llm, "apredict"):
                    result = await llm.apredict(prompt_text)
                elif hasattr(llm, "predict"):
                    result = llm.predict(prompt_text)
                else:
                    raise RuntimeError("No invocation method available on chain or llm.")
        except Exception as e:
            exc = e
            tb = traceback.format_exc()
            print("ERROR during LLM invocation:", repr(e))
            print(tb)

        # Normalize result
        text = ""
        if isinstance(result, dict):
            for key in ("text", "output_text", "answer", "response"):
                if key in result:
                    text = result[key]
                    break
            if not text:
                text = " ".join([str(v) for v in result.values()])
        elif isinstance(result, str):
            text = result
        elif result is None and exc:
            text = f"Internal error: {str(exc)}"
        else:
            text = str(result)

        duration = time.time() - start_ts
        print(f"INFO: chat_ai_async finished in {duration:.3f}s; question len={len(question)}; docs={len(docs)}")
        return {"message": text.strip()}

    except Exception as e:
        tb = traceback.format_exc()
        print("UNHANDLED ERROR in chat_ai_async:", repr(e))
        print(tb)
        return {"message": f"Internal error: {str(e)}"}


# --- API endpoints ---
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/chat")
async def chat(user_id: str, option: str):
    if option == "main":
        return JSONResponse(
            {
                "message": "Hello! Welcome to ApnaBazzar! How may I help you today?",
                "options": ["Order Related", "Product Related", "Others"],
            }
        )
    if option == "Order Related":
        return JSONResponse({"message": "Please choose an option related to your orders:", "options": ["Recent Order", "All Orders", "Track Order", "Back"]})
    if option == "Product Related":
        return JSONResponse({"message": "Need help with products? Select an option below:", "options": ["Request Product", "Back"]})
    if option == "Others":
        return JSONResponse({"message": "Chat with AI Assistant", "options": ["Chat with AI Assistant", "Back"]})

    if option == "Recent Order":
        orders_snip = fetch_user_orders_snippet(user_id, limit=1)
        return JSONResponse({"message": orders_snip or "No data found"})

    if option == "All Orders":
        orders_snip = fetch_user_orders_snippet(user_id, limit=5)
        return JSONResponse({"message": orders_snip or "No data found"})

    if option == "Track Order":
        orders_snip = fetch_user_orders_snippet(user_id, limit=1)
        return JSONResponse({"message": orders_snip or "No data found"})

    if option == "Request Product":
        return JSONResponse({"message": "Send us the product name you want to request (not available on site).", "options": ["Back"]})

    if option == "Chat with AI Assistant":
        return JSONResponse({"message": "You’re now connected to the AI Assistant. Please type your question below:", "options": ["Back"]})

    if option == "Back":
        return JSONResponse(await chat(user_id, "main"))

    return JSONResponse({"message": "Invalid option. Try again.", "options": ["Back"]})


@app.get("/chat/ai")
async def chat_ai_endpoint(user_id: str, question: str):
    resp = await chat_ai_async(user_id, question)
    return JSONResponse(resp)


# --- Debug endpoints ---
@app.get("/debug/search")
def debug_search(query: str):
    try:
        docs = cached_search(query)
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse({"ok": False, "error": repr(e), "trace": tb}, status_code=500)
    snippets = [{"score": getattr(d, "score", None), "text": d.page_content[:800]} for d in docs]
    return JSONResponse({"ok": True, "count": len(snippets), "snippets": snippets})


@app.get("/debug/llm")
async def debug_llm(prompt: str = "Hello"):
    try:
        llm = get_llm()
        if hasattr(llm, "apredict"):
            out = await llm.apredict(prompt)
        elif hasattr(llm, "predict"):
            out = llm.predict(prompt)
        else:
            raise RuntimeError("LLM has no predict/apredict method.")
        return JSONResponse({"ok": True, "result": out})
    except Exception as e:
        tb = traceback.format_exc()
        print("DEBUG LLM error:", repr(e))
        print(tb)
        return JSONResponse({"ok": False, "error": repr(e), "trace": tb}, status_code=500)
