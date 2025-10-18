from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import os
import gc

# -------------------------------------------------
# FASTAPI SETUP
# -------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
MONGODB_URI = os.environ.get("MONGODB_URI")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")

# -------------------------------------------------
# LIGHTWEIGHT DATA FETCH (NO LARGE DATAFRAMES)
# -------------------------------------------------
def fetch_data():
    client = MongoClient(MONGODB_URI)
    db = client["ECommerce"]

    products = list(db["products"].find(
        {"isActive": True},
        {"name": 1, "category": 1, "price": 1, "description": 1, "_id": 1}
    ))
    users = list(db["users"].find({}, {"_id": 1, "orders": 1}))

    client.close()

    # Convert minimal data
    product_lines = [
        f"{p.get('name', '')}, {p.get('category', '')}, {p.get('price', '')}, {p.get('description', '')}, id={p.get('_id')}"
        for p in products
    ]
    order_lines = [
        f"user={u.get('_id')}, orders={u.get('orders', [])}" for u in users
    ]

    gc.collect()
    return "\n".join(product_lines), "\n".join(order_lines)

product_text, order_text = fetch_data()

# -------------------------------------------------
# FAISS + EMBEDDINGS (MINIMAL)
# -------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.create_documents([product_text])

# embeddings = SentenceTransformerEmbeddings(
#     model_name="all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={"normalize_embeddings": True}
# )
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    

gc.collect()

# -------------------------------------------------
# LIGHTWEIGHT LLM CHAIN (ONE INSTANCE)
# -------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

prompt = PromptTemplate(
    template=(
        "You are a concise e-commerce assistant. "
        "Answer ONLY from the context. Keep replies ≤2 lines.\n\n"
        "If not found, say 'No data found.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    ),
    input_variables=["context", "question"]
)

chain = LLMChain(llm=llm, prompt=prompt)

# -------------------------------------------------
# AI CHAT ENDPOINT
# -------------------------------------------------
@app.get("/chat/ai")
async def chat_ai(user_id: str, question: str):
    try:
        if not question.strip():
            return JSONResponse({"message": "No query provided."})

        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local("faiss_index")

        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])

        result = await chain.ainvoke({"context": context, "question": question})
        answer = result.get("text", str(result))

        del docs, context, result
        gc.collect()
        return JSONResponse({"message": answer})

    except Exception as e:
        gc.collect()
        return JSONResponse({"message": f"Internal error: {e}"})

# -------------------------------------------------
# BASIC MENU FLOW (ULTRA LIGHT)
# -------------------------------------------------
@app.get("/chat")
async def chat(user_id: str, option: str):
    if option == "main":
        return JSONResponse({
            "message": "👋 Welcome to ApnaBazzar! How may I help you?",
            "options": ["Order Related", "Product Related", "Others"]
        })

    if option == "Order Related":
        return JSONResponse({"message": "Order options:", "options": ["Recent", "All", "Back"]})

    if option == "Product Related":
        return JSONResponse({"message": "Product options:", "options": ["Request Product", "Back"]})

    if option == "Others":
        return JSONResponse({"message": "Chat with AI Assistant below 👇", "options": ["Chat with AI", "Back"]})

    if option == "Back":
        return await chat(user_id, "main")

    return JSONResponse({"message": "Invalid option.", "options": ["Back"]})

