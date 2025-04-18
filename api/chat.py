import os
import json
import logging
import re
import sys
from typing import Optional, List, Dict, Tuple, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from datetime import datetime
import urllib.parse
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')
MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')

# Configure Gemini API
model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {str(e)}")

# API configuration
FOURSQUARE_BASE_URL = "https://api.foursquare.com/v3/places/search"
FOURSQUARE_PLACE_DETAILS_URL = "https://api.foursquare.com/v3/places/"
FOURSQUARE_PHOTOS_URL = "https://api.foursquare.com/v3/places/{}/photos"
MAPTILER_GEOCODING_URL = "https://api.maptiler.com/geocoding"

# API Headers
FOURSQUARE_HEADERS = {
    "Authorization": FOURSQUARE_API_KEY,
    "accept": "application/json"
} if FOURSQUARE_API_KEY else {}

MAPTILER_HEADERS = {
    "accept": "application/json"
}

# Models
class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None

class Venue(BaseModel):
    name: str
    type: str
    address: str
    rating: Optional[float]
    price: Optional[str]
    capacity: Optional[int]
    image: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    description: Optional[str]
    amenities: Optional[List[str]]
    hours: Optional[Dict[str, str]]
    contact: Optional[Dict[str, str]]
    relevance_score: Optional[float]

class ChatResponse(BaseModel):
    response: str
    venues: Optional[List[dict]] = None
    error: Optional[str] = None

# Utility Functions
def retry_with_backoff(func, max_retries=3, initial_delay=1):
    """Retry a function with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay)
            delay *= 2

def clean_gemini_response(text: str) -> str:
    """Clean Gemini response by removing markdown formatting."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    return text.strip()

def get_geocode(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude and longitude for a location using MapTiler."""
    if not MAPTILER_API_KEY:
        return None, None
        
    try:
        def geocode_request():
            response = requests.get(
                f"{MAPTILER_GEOCODING_URL}/{urllib.parse.quote(location)}.json",
                params={"key": MAPTILER_API_KEY},
                headers=MAPTILER_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(geocode_request)
        
        if data.get('features'):
            coords = data['features'][0]['center']
            return coords[1], coords[0]
        return None, None
    except Exception as e:
        logger.error(f"Geocoding error for location '{location}': {str(e)}")
        return None, None

def get_venue_photos(fsq_id: str) -> List[str]:
    """Get photos for a venue from Foursquare."""
    if not FOURSQUARE_API_KEY:
        return []
        
    try:
        def photos_request():
            response = requests.get(
                FOURSQUARE_PHOTOS_URL.format(fsq_id),
                headers=FOURSQUARE_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(photos_request)
        return [f"{photo['prefix']}original{photo['suffix']}" 
               for photo in data.get('items', [])[:3] 
               if 'prefix' in photo and 'suffix' in photo]
    except Exception as e:
        logger.error(f"Error getting venue photos for {fsq_id}: {str(e)}")
        return []

def get_venue_details(fsq_id: str) -> Dict[str, Any]:
    """Get detailed information about a venue from Foursquare."""
    if not FOURSQUARE_API_KEY:
        return {}
        
    try:
        def details_request():
            response = requests.get(
                f"{FOURSQUARE_PLACE_DETAILS_URL}{fsq_id}",
                headers=FOURSQUARE_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(details_request)
        return {
            'capacity': data.get('capacity', 0),
            'description': data.get('description', ''),
            'amenities': data.get('amenities', []),
            'hours': data.get('hours', {}),
            'contact': data.get('contact', {}),
            'photos': get_venue_photos(fsq_id)
        }
    except Exception as e:
        logger.error(f"Error getting venue details for {fsq_id}: {str(e)}")
        return {}

def process_venues(raw_venues: List[dict], event_type: str, keywords: List[str]) -> List[dict]:
    """Process raw venue data into standardized format."""
    processed_venues = []
    
    for venue in raw_venues[:10]:  # Limit to first 10 venues
        try:
            fsq_id = venue.get('fsq_id')
            if not fsq_id:
                continue
                
            details = get_venue_details(fsq_id)
            
            processed_venue = {
                'name': venue.get('name', 'Unknown Venue'),
                'type': venue.get('categories', [{}])[0].get('name', 'Venue'),
                'address': venue.get('location', {}).get('formatted_address', 'Address not available'),
                'rating': venue.get('rating'),
                'price': venue.get('price'),
                'capacity': details.get('capacity'),
                'image': details['photos'][0] if details.get('photos') else None,
                'latitude': venue.get('geocodes', {}).get('main', {}).get('latitude'),
                'longitude': venue.get('geocodes', {}).get('main', {}).get('longitude'),
                'description': details.get('description'),
                'amenities': details.get('amenities'),
                'hours': details.get('hours'),
                'contact': details.get('contact'),
                'relevance_score': calculate_relevance_score(venue, event_type, keywords)
            }
            
            processed_venues.append(processed_venue)
        except Exception as e:
            logger.error(f"Error processing venue {venue.get('name')}: {str(e)}")
            continue
    
    return sorted(processed_venues, key=lambda x: x.get('relevance_score', 0), reverse=True)

def calculate_relevance_score(venue: dict, event_type: str, keywords: List[str]) -> float:
    """Calculate a relevance score for a venue."""
    score = 0.0
    score += venue.get('rating', 0) * 0.2
    
    venue_categories = [cat.get('name', '').lower() for cat in venue.get('categories', [])]
    for keyword in keywords:
        if any(keyword.lower() in cat for cat in venue_categories):
            score += 0.3
    
    description = venue.get('description', '').lower()
    for keyword in keywords:
        if keyword.lower() in description:
            score += 0.2
    
    name = venue.get('name', '').lower()
    for keyword in keywords:
        if keyword.lower() in name:
            score += 0.3
    
    return min(score, 1.0)

def extract_event_details(message: str) -> Dict[str, Any]:
    """Extract event details from the user message."""
    message = message.lower()
    details = {
        'event_type': 'general',
        'location': None,
        'capacity': None,
        'budget': None,
        'specific_requirements': [],
        'keywords': []
    }

    if model:
        try:
            prompt = f"""Analyze this venue search request and extract key information:
            Message: {message}
            
            Please provide a JSON response with:
            1. event_type (business, wedding, social, conference, sports, dining, accommodation)
            2. location (city or area)
            3. capacity (number of people)
            4. budget (in dollars)
            5. specific_requirements (list of specific needs)
            6. keywords (list of relevant keywords for the search)
            
            Format: {{"event_type": "...", "location": "...", "capacity": number, "budget": number, "specific_requirements": [...], "keywords": [...]}}
            
            If any field is not found, use null or an empty list."""
            
            def gemini_request():
                response = model.generate_content(prompt)
                return response.text

            gemini_response = retry_with_backoff(gemini_request)
            try:
                gemini_data = json.loads(gemini_response)
                details.update(gemini_data)
            except json.JSONDecodeError:
                logger.error("Failed to parse Gemini response")
        except Exception as e:
            logger.error(f"Error using Gemini for analysis: {str(e)}")

    if not details['location']:
        location_match = re.search(r'(in|at|near|around|close to|within)\s+([^,.!?]+)', message)
        if location_match:
            details['location'] = location_match.group(2).strip()

    if not details['capacity']:
        capacity_match = re.search(r'(\d+)\s*(people|guests|attendees|capacity|persons|individuals)', message)
        if capacity_match:
            details['capacity'] = int(capacity_match.group(1))

    if not details['budget']:
        budget_match = re.search(r'(\$|₹|€|£)?\s*(\d+)\s*(k|thousand|K)?', message)
        if budget_match:
            amount = int(budget_match.group(2))
            if budget_match.group(3):
                amount *= 1000
            details['budget'] = amount

    if not details['keywords']:
        words = re.findall(r'\b\w{4,}\b', message)
        details['keywords'] = [word for word in words if word not in ['find', 'looking', 'for', 'venue', 'place']]

    return details

def search_foursquare_venues(location: str, query: str, limit: int = 10) -> List[dict]:
    """Search for venues using Foursquare API."""
    if not FOURSQUARE_API_KEY:
        return []
        
    try:
        params = {
            'query': query,
            'near': location,
            'limit': limit,
            'fields': 'name,location,rating,price,fsq_id,geocodes,photos,categories,description'
        }
        
        response = requests.get(
            FOURSQUARE_BASE_URL,
            headers=FOURSQUARE_HEADERS,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        return response.json().get('results', [])
    except Exception as e:
        logger.error(f"Foursquare API error for query '{query}' in '{location}': {str(e)}")
        return []

def get_gemini_response(message: str) -> str:
    """Get response from Gemini API."""
    if not GEMINI_API_KEY:
        return "AI service is currently unavailable."
        
    try:
        prompt = {
            "contents": [{
                "parts": [{
                    "text": f"User: {message}\nAssistant:"
                }]
            }]
        }
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json=prompt,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        return clean_gemini_response(result['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return "I'm having trouble processing your request. Please try again."

# Routes
@app.get("/")
async def read_root():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "templates/index.html")
        if os.path.exists(file_path):
            return FileResponse(file_path)
        return JSONResponse(content={"message": "Welcome to Venue Finder API"})
    except Exception as e:
        logger.error(f"Error serving root: {str(e)}")
        return JSONResponse(content={"message": "Welcome to Venue Finder API"})

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "gemini": bool(GEMINI_API_KEY),
            "foursquare": bool(FOURSQUARE_API_KEY),
            "maptiler": bool(MAPTILER_API_KEY)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/debug")
async def debug_info():
    return {
        "python_version": sys.version,
        "environment_vars": {
            "GEMINI_API_KEY": bool(GEMINI_API_KEY),
            "FOURSQUARE_API_KEY": bool(FOURSQUARE_API_KEY),
            "MAPTILER_API_KEY": bool(MAPTILER_API_KEY)
        },
        "file_structure": {
            "static_exists": os.path.exists("static"),
            "templates_exists": os.path.exists("templates")
        }
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message or len(request.message.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Please provide more details about the venue you're looking for."
            )

        event_details = extract_event_details(request.message)
        location = event_details.get('location')
        
        if not location:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "response": "Please specify a location in your request (e.g., 'Find venues in New York')."
                }
            )

        lat, lng = get_geocode(location)
        if not lat or not lng:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "response": f"I couldn't find the location '{location}'. Please try being more specific."
                }
            )

        query_mapping = {
            'wedding': 'wedding venue',
            'business': 'conference center',
            'conference': 'conference center',
            'sports': 'sports venue',
            'dining': 'restaurant',
            'accommodation': 'hotel',
            'social': 'event space'
        }
        
        query = query_mapping.get(event_details['event_type'], 'venue')
        if event_details['keywords']:
            query = ' '.join([query] + event_details['keywords'][:2])

        raw_venues = search_foursquare_venues(location, query)
        if not raw_venues:
            return JSONResponse(
                content={
                    "success": True,
                    "response": f"I couldn't find any {query} venues in {location}. Try broadening your search criteria.",
                    "venues": []
                }
            )

        venues = process_venues(raw_venues, event_details['event_type'], event_details['keywords'])
        
        try:
            response = get_gemini_response(
                f"Generate a friendly response about finding {len(venues)} {query} venues in {location}"
            )
        except Exception:
            response = f"I found {len(venues)} venues in {location} that match your criteria."

        return JSONResponse(
            content={
                "success": True,
                "response": response,
                "venues": venues[:10]
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again."
        )

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found."}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )

# Mount static files if they exist
static_path = os.path.join(os.path.dirname(__file__), "static") if os.path.exists(os.path.join(os.path.dirname(__file__), "static")) else None
if static_path:
    app.mount("/static", StaticFiles(directory=static_path), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
