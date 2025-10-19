# app.py
import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from pymongo import MongoClient
from bson.objectid import ObjectId

# langchain imports (match your installed packages)
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration (use Render environment variables) ---
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

# --- Helpers: lazy singletons to save memory ---


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI environment variable not set")
    return MongoClient(MONGODB_URI)


@lru_cache(maxsize=1)
def get_embeddings():
    # Cached and instanced once per process
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = get_embeddings()
    # Load local index created offline
    try:
        vs = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}")
    return vs


@lru_cache(maxsize=1)
def get_llm():
    # Keep LLM instance once per process
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# Small cache for similarity queries; keep size bounded
@lru_cache(maxsize=128)
def cached_search(query: str):
    vs = get_vector_store()
    return vs.similarity_search(query, k=5)


# Prompt template (shared)
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


# --- Lightweight DB snippets on demand (avoid loading full collections) ---
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

    # Try to convert to ObjectId; if fails, try to query by stored string _id
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


# --- Core chat function ---
async def chat_ai_async(user_id: str, question: str):
    if not question:
        return {"message": "No query found for user.", "options": ["Back"]}

    try:
        docs = cached_search(question)
        context = "\n".join([d.page_content for d in docs]) if docs else ""
        products_snip = fetch_product_snippet(limit=10)
        orders_snip = fetch_user_orders_snippet(user_id, limit=5)

        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=PROMPT)

        result = await chain.ainvoke(
            {"context": context, "question": question, "products": products_snip, "orders": orders_snip}
        )

        text = result["text"] if isinstance(result, dict) and "text" in result else str(result)
        return {"message": text.strip()}

    except Exception as e:
        return {"message": f"Internal error: {str(e)}"}


# --- API routes (small responses only) ---
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
