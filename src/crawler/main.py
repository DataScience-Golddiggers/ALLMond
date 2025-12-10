from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup

app = FastAPI()

# Hardcoded UnivPM URLs
UNIVPM_URLS = [
    "https://www.univpm.it/Entra/",
    "https://www.univpm.it/Entra/Ateneo",
    "https://www.univpm.it/Entra/Didattica",
    "https://www.univpm.it/Entra/Ricerca"
]

class CrawlRequest(BaseModel):
    urls: Optional[List[str]] = None

class EbayProductRequest(BaseModel):
    url: str

@app.post("/crawl")
async def crawl(request: CrawlRequest):
    urls_to_crawl = request.urls if request.urls else UNIVPM_URLS
    
    results = []
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        for url in urls_to_crawl:
            try:
                result = await crawler.arun(url=url)
                # result.markdown contains the content converted to markdown (stripping HTML)
                results.append({
                    "url": url,
                    "content": result.markdown,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "success": False
                })
                
    return {"results": results}

@app.post("/crawl/ebay")
async def crawl_ebay(request: EbayProductRequest):
    url = request.url
    async with AsyncWebCrawler(verbose=True) as crawler:
        try:
            result = await crawler.arun(url=url)
            
            if not result.markdown: # Basic check if crawl worked
                 raise HTTPException(status_code=400, detail="Failed to crawl URL or empty content")
            
            soup = BeautifulSoup(result.html, 'html.parser')
            
            # Title
            title_tag = soup.select_one('h1.x-item-title__mainTitle') or soup.select_one('#itemTitle')
            title = title_tag.get_text(strip=True).replace('Details about', '').strip() if title_tag else "Title not found"
            
            # Price
            price_tag = soup.select_one('.x-price-primary') or soup.select_one('#prcIsum')
            price = price_tag.get_text(strip=True) if price_tag else "Price not found"
            
            # Description
            # Using markdown as fallback/primary source for description text
            description = result.markdown[:5000] 
            
            # Reviews
            reviews = []
            # Try multiple selectors for reviews
            review_elements = soup.select('.fdbk-detail-list__entry') or soup.select('.ebay-review-section .review-item')
            
            for el in review_elements[:5]: # Limit to 5
                text_el = el.select_one('.fdbk-container__details-content') or el.select_one('.review-content')
                if text_el:
                    reviews.append(text_el.get_text(strip=True))
            
            reviews_message = "Nessuna recensione disponibile" if not reviews else None
            
            return {
                "title": title,
                "price": price,
                "description": description,
                "reviews": reviews,
                "reviews_message": reviews_message,
                "url": url
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
