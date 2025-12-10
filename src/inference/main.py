import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
import httpx
import asyncio
from datetime import datetime, timedelta
# LangChain imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Add /app to path to import utils
sys.path.append("/app")
from utils.text_preprocessing import TextPreprocessor
from utils.database import init_db, get_db, Product, Review
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

app = FastAPI()

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
CRAWLER_URL = os.getenv("CRAWLER_URL", "http://crawler:8001")
SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL", "http://sentiment-service:8002")
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 24))
MODEL_PATH = "/app/models/logistic_regression.pkl"
VECTORIZER_PATH = "/app/models/vectorizer.pkl"

# Model names
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen3:0.6b")
QA_MODEL = os.getenv("QA_MODEL", "qwen3:1.7b")

# Models DTOs
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    relevant: bool
    context_used: bool

class ProductRequest(BaseModel):
    url: str

class ProductResponse(BaseModel):
    title: str
    price: str
    description: str
    summary: str
    reviews: List[dict]
    sentiment_breakdown: dict
    cached: bool

# Global variables
classifier = None
vectorizer = None
preprocessor = None
llm_summary = None
llm_qa = None

@app.on_event("startup")
async def startup_event():
    global classifier, vectorizer, preprocessor, llm_summary, llm_qa
    
    # Initialize DB
    try:
        await init_db()
        print("Database initialized.")
    except Exception as e:
        print(f"Error initializing database: {e}")

    # 1. Load Scikit-Learn Models (Classification)
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            classifier = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("Classifier and Vectorizer loaded successfully.")
        else:
            print(f"Models not found at {MODEL_PATH} or {VECTORIZER_PATH}. Inference will fail.")
    except Exception as e:
        print(f"Error loading models: {e}")

    # 2. Initialize Preprocessor
    preprocessor = TextPreprocessor(language="italian")
    
    # 3. Initialize LangChain Ollama Chat Models
    try:
        print(f"Initializing Ollama with URL: {OLLAMA_URL}")
        llm_summary = ChatOllama(
            base_url=OLLAMA_URL,
            model=SUMMARY_MODEL,
            temperature=0.2
        )
        llm_qa = ChatOllama(
            base_url=OLLAMA_URL,
            model=QA_MODEL,
            temperature=0.7
        )
        print("LangChain Ollama instances initialized.")
    except Exception as e:
        print(f"Error initializing LangChain Ollama: {e}")


def classify_relevance(question: str) -> bool:
    """Returns True if the question is relevant using the loaded classifier."""
    if not classifier or not vectorizer:
        # If classifier is missing, we assume relevant to allow usage.
        return True
        
    cleaned_question = preprocessor.preprocess(question)
    X = vectorizer.transform([cleaned_question])
    prediction = classifier.predict(X)[0]
    return bool(prediction == 1)

async def crawl_content() -> str:
    """Fetches content from the crawler service asynchronously."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            print(f"Richiesta al crawler inviata a {CRAWLER_URL}/crawl...")
            response = await client.post(f"{CRAWLER_URL}/crawl", json={"urls": []})
            
            if response.status_code != 200:
                print(f"Crawler service returned status {response.status_code}: {response.text}")
                return ""
                
            crawl_data = response.json()
            # Default to empty list if results key is missing or None
            results = crawl_data.get("results") or []
            
            full_text = ""
            for i, res in enumerate(results):
                try:
                    url = res.get("url", "unknown_url")
                    success = res.get("success", False)
                    
                    if success:
                        content = res.get("content")
                        if content is None:
                            print(f"Warning: Result {i} for {url} success=True but content is None")
                            continue
                            
                        if not isinstance(content, str):
                            print(f"Warning: Result {i} for {url} content is not string (type: {type(content)})")
                            content = str(content)
                            
                        full_text += content + "\n\n"
                    else:
                        error_msg = res.get("error", "Unknown error")
                        print(f"Failed to crawl specific URL: {url} - Error: {error_msg}")
                except Exception as inner_e:
                    print(f"Error processing result {i}: {inner_e}")
            
            if not full_text:
                print("Crawler returned no text content.")
                
            return full_text
            
        except Exception as e:
            # Print full stack trace or detailed error to debug 'NoneType' issues
            import traceback
            traceback.print_exc()
            print(f"Crawler request failed completely: {e}")
            return ""

async def summarize_description(text: str) -> str:
    if not text: return ""
    truncated = text[:5000]
    template = """Summarize this product description focusing on features and specs: {text}"""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm_summary | StrOutputParser()
    try:
        return await chain.ainvoke({"text": truncated})
    except Exception as e:
        print(f"Summarization error: {e}")
        return text[:500] + "..."

async def analyze_sentiment_remote(texts: List[str]) -> List[dict]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{SENTIMENT_SERVICE_URL}/analyze", json={"texts": texts})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Sentiment service error: {response.text}")
                return []
        except Exception as e:
            print(f"Sentiment service connection error: {e}")
            return []

@app.post("/api/product", response_model=ProductResponse)
async def get_product_info(request: ProductRequest):
    url = request.url
    
    # 1. Check DB
    async for session in get_db():
        stmt = select(Product).where(Product.ebay_url == url)
        result = await session.execute(stmt)
        product = result.scalars().first()
        
        if product:
            # Check TTL
            if product.crawled_at.replace(tzinfo=None) > datetime.now() - timedelta(hours=CACHE_TTL_HOURS):
                # Fetch reviews
                stmt_rev = select(Review).where(Review.product_id == product.id)
                res_rev = await session.execute(stmt_rev)
                reviews = res_rev.scalars().all()
                
                reviews_data = [
                    {"text": r.text, "sentiment": r.sentiment, "confidence": r.confidence} 
                    for r in reviews
                ]
                
                # Calculate breakdown
                breakdown = {"Positive": 0, "Negative": 0, "Neutral": 0}
                for r in reviews:
                    if r.sentiment in breakdown:
                        breakdown[r.sentiment] += 1
                
                return {
                    "title": product.title,
                    "price": product.price,
                    "description": product.description,
                    "summary": product.summary,
                    "reviews": reviews_data,
                    "sentiment_breakdown": breakdown,
                    "cached": True
                }
            else:
                # Expired, we will update it
                pass

    # 2. Crawl
    async with httpx.AsyncClient(timeout=60.0) as client:
        crawl_res = await client.post(f"{CRAWLER_URL}/crawl/ebay", json={"url": url})
        if crawl_res.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to crawl eBay product")
        
        data = crawl_res.json()
        
    # 3. Summarize & Sentiment (Parallel)
    description = data.get("description", "")
    reviews_text = data.get("reviews", [])
    
    # Summarize
    summary_task = asyncio.create_task(summarize_description(description))
    
    # Sentiment
    sentiment_task = asyncio.create_task(analyze_sentiment_remote(reviews_text)) if reviews_text else asyncio.create_task(asyncio.sleep(0, result=[]))
    
    summary, sentiment_results = await asyncio.gather(summary_task, sentiment_task)
    
    # 4. Save to DB
    async for session in get_db():
        # Check if exists (to update)
        stmt = select(Product).where(Product.ebay_url == url)
        result = await session.execute(stmt)
        existing_product = result.scalars().first()
        
        if existing_product:
            existing_product.title = data.get("title")
            existing_product.price = data.get("price")
            existing_product.description = description
            existing_product.summary = summary
            existing_product.crawled_at = datetime.now()
            
            # Delete old reviews
            await session.execute(delete(Review).where(Review.product_id == existing_product.id))
            
            product_id = existing_product.id
        else:
            new_product = Product(
                ebay_url=url,
                title=data.get("title"),
                price=data.get("price"),
                description=description,
                summary=summary
            )
            session.add(new_product)
            await session.flush()
            product_id = new_product.id
            
        # Add reviews
        for res in sentiment_results:
            new_review = Review(
                product_id=product_id,
                text=res['text'],
                sentiment=res['sentiment'],
                confidence=res['confidence']
            )
            session.add(new_review)
            
        await session.commit()
        
    # 5. Return
    breakdown = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in sentiment_results:
        if r['sentiment'] in breakdown:
            breakdown[r['sentiment']] += 1
            
    return {
        "title": data.get("title"),
        "price": data.get("price"),
        "description": description,
        "summary": summary,
        "reviews": sentiment_results,
        "sentiment_breakdown": breakdown,
        "cached": False
    }

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # 1. Classification (Relevance Check)
    # We still check if the question is about the University, but we don't judge the content yet.
    try:
        is_relevant = classify_relevance(request.question)
        if not is_relevant:
            return AskResponse(
                answer="La tua domanda non sembra essere pertinente con l\'argomento trattato (Università Politecnica delle Marche).",
                relevant=False,
                context_used=False
            )
    except Exception as e:
        print(f"Classification error: {e}")
        # Proceed if check fails

    # 2. Retrieval (Crawling)
    # Now correctly awaited and async
    raw_content = await crawl_content()
    
    # If crawling fails completely or returns nothing, we can't do RAG.
    if not raw_content:
        return AskResponse(
            answer="Non sono riuscito a recuperare le informazioni dai siti web dell\'Università.",
            relevant=True,
            context_used=False
        )

    # 3. LangChain Processing
    
    # A. Summarization Chain
    # We truncate input to avoid context window explosion before summarization if needed.
    truncated_content = raw_content[:15000] 
    
    summary_template = """Sei un assistente utile. Riassumi il seguente testo estratto dal sito web dell\'Università.
Mantieni tutte le informazioni rilevanti, date, scadenze e contatti. Rimuovi solo le parti puramente tecniche o di navigazione (menu, cookie policy, ecc).

Testo:
{text}

Riassunto:"""
    
    summary_prompt = PromptTemplate.from_template(summary_template)
    summarize_chain = summary_prompt | llm_summary | StrOutputParser()

    try:
        # Execute Summarization
        print("Inizio riassunto...")
        summary_text = await summarize_chain.ainvoke({"text": truncated_content})
        print("Riassunto completato.")
    except Exception as e:
        print(f"Summarization error: {e}")
        summary_text = truncated_content[:3000] # Fallback to raw text

    # B. QA Chain
    qa_template = """Sei un assistente intelligente che risponde alla domanda dell\'utente basandoti SOLO sul contesto fornito qui sotto.

                    Contesto:
                    {context}

                    Domanda: {question}

                    Risposta:"""

    qa_prompt = PromptTemplate.from_template(qa_template)
    qa_chain = qa_prompt | llm_qa | StrOutputParser()

    try:
        # Execute QA
        print("Generazione risposta...")
        answer_text = await qa_chain.ainvoke({
            "context": summary_text,
            "question": request.question
        })
    except Exception as e:
        print(f"QA error: {e}")
        answer_text = "Si è verificato un errore durante la generazione della risposta."

    return AskResponse(
        answer=answer_text,
        relevant=True,
        context_used=True
    )

@app.get("/health")
def health():
    return {"status": "ok", "ollama_url": OLLAMA_URL}
