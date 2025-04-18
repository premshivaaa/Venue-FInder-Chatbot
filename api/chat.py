import os
import json
import logging
import re
from typing import Optional, List, Dict, Tuple
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

# Mount static files
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    model = None

# Foursquare API configuration
FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')
FOURSQUARE_BASE_URL = "https://api.foursquare.com/v3/places/search"
FOURSQUARE_PLACE_DETAILS_URL = "https://api.foursquare.com/v3/places/"
FOURSQUARE_PHOTOS_URL = "https://api.foursquare.com/v3/places/{}/photos"

# MapTiler API configuration
MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')
MAPTILER_GEOCODING_URL = "https://api.maptiler.com/geocoding"

# API Headers
FOURSQUARE_HEADERS = {
    "Authorization": FOURSQUARE_API_KEY,
    "accept": "application/json"
}

MAPTILER_HEADERS = {
    "accept": "application/json"
}

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
    # Remove markdown bold syntax
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove other markdown syntax
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    return text.strip()

def get_geocode(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude and longitude for a location using MapTiler."""
    try:
        def geocode_request():
            response = requests.get(
                f"{MAPTILER_GEOCODING_URL}/{urllib.parse.quote(location)}.json",
                params={"key": MAPTILER_API_KEY},
                headers=MAPTILER_HEADERS
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(geocode_request)
        
        if data.get('features'):
            coords = data['features'][0]['center']
            return coords[1], coords[0]  # latitude, longitude
        return None, None
    except Exception as e:
        logger.error(f"Geocoding error: {str(e)}")
        return None, None

def get_venue_photos(fsq_id: str) -> List[str]:
    """Get photos for a venue from Foursquare."""
    try:
        def photos_request():
            response = requests.get(
                FOURSQUARE_PHOTOS_URL.format(fsq_id),
                headers=FOURSQUARE_HEADERS
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(photos_request)
        
        photos = []
        for photo in data.get('items', []):
            if 'prefix' in photo and 'suffix' in photo:
                photos.append(f"{photo['prefix']}original{photo['suffix']}")
        
        return photos
    except Exception as e:
        logger.error(f"Error getting venue photos: {str(e)}")
        return []

def get_venue_details(fsq_id: str) -> Dict:
    """Get detailed information about a venue from Foursquare."""
    try:
        def details_request():
            response = requests.get(
                f"{FOURSQUARE_PLACE_DETAILS_URL}{fsq_id}",
                headers=FOURSQUARE_HEADERS
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(details_request)
        
        # Get photos
        photos = get_venue_photos(fsq_id)
        
        return {
            'capacity': data.get('capacity', 0),
            'description': data.get('description', ''),
            'amenities': data.get('amenities', []),
            'hours': data.get('hours', {}),
            'contact': data.get('contact', {}),
            'photos': photos
        }
    except Exception as e:
        logger.error(f"Error getting venue details: {str(e)}")
        return {}

def calculate_relevance_score(venue: dict, event_type: str, keywords: List[str]) -> float:
    """Calculate a relevance score for a venue based on the search criteria."""
    score = 0.0
    
    # Base score from rating
    score += venue.get('rating', 0) * 0.2
    
    # Category match
    venue_categories = [cat.lower() for cat in venue.get('categories', [])]
    for keyword in keywords:
        if any(keyword in cat for cat in venue_categories):
            score += 0.3
    
    # Description match
    description = venue.get('description', '').lower()
    for keyword in keywords:
        if keyword in description:
            score += 0.2
    
    # Name match
    name = venue.get('name', '').lower()
    for keyword in keywords:
        if keyword in name:
            score += 0.3
    
    return min(score, 1.0)  # Normalize to 0-1 range

def extract_event_details(message: str) -> Dict:
    """Extract event details using NLP techniques."""
    message = message.lower()
    details = {
        'event_type': 'general',
        'location': None,
        'capacity': None,
        'budget': None,
        'specific_requirements': [],
        'keywords': []
    }

    # Use Gemini for initial analysis if available
    if model:
        try:
            prompt = f"""Analyze this venue search request and extract key information:
            Message: {message}
            
            Please provide a JSON response with:
            1. event_type (business, sports, wedding, social, graduation, exhibition, dining, accommodation, entertainment)
            2. location (city or area)
            3. capacity (number of people)
            4. budget (in dollars)
            5. specific_requirements (list of specific needs)
            6. keywords (list of relevant keywords for the search)
            
            Format: {{"event_type": "...", "location": "...", "capacity": number, "budget": number, "specific_requirements": [...], "keywords": [...]}}
            
            If any field is not found, use null or an empty list. Never return an error."""
            
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

    # Fallback to regex if Gemini fails or is not available
    if not details['location']:
        location_pattern = r'(in|at|near|around|close to|within)\s+([^,.!?]+)'
        location_match = re.search(location_pattern, message)
        if location_match:
            details['location'] = location_match.group(2).strip()

    if not details['capacity']:
        capacity_pattern = r'(\d+)\s*(people|guests|attendees|capacity|persons|individuals)'
        capacity_match = re.search(capacity_pattern, message)
        if capacity_match:
            details['capacity'] = int(capacity_match.group(1))

    if not details['budget']:
        budget_pattern = r'(\$|₹|€|£)?\s*(\d+)\s*(k|thousand|K)?'
        budget_match = re.search(budget_pattern, message)
        if budget_match:
            amount = int(budget_match.group(2))
            if budget_match.group(3):
                amount *= 1000
            details['budget'] = amount

    # Extract keywords from the message
    if not details['keywords']:
        words = message.split()
        details['keywords'] = [word for word in words if len(word) > 3]

    return details

def search_foursquare_venues(location: str, query: str, limit: int = 10) -> List[dict]:
    """Search for venues using Foursquare API."""
    try:
        def venue_request():
            params = {
                'query': query,
                'near': location,
                'limit': limit,
                'fields': 'name,location,rating,price,fsq_id,geocodes,photos,categories,description'
            }
            
            response = requests.get(
                "https://api.foursquare.com/v3/places/search",
                headers=FOURSQUARE_HEADERS,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(venue_request)
        return data.get('results', [])
    except Exception as e:
        logger.error(f"Foursquare API error: {str(e)}")
        return []

def get_gemini_response(message: str) -> str:
    """Get response from Gemini API."""
    try:
        def gemini_request():
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
            return response.json()

        result = retry_with_backoff(gemini_request)
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return "I'm having trouble processing your request. Please try again."

def generate_helpful_response(query: str, location: str, capacity: Optional[int], budget: Optional[int], venues: List[dict]) -> str:
    """Generate a helpful response based on the search results."""
    try:
        if not venues:
            return f"""I couldn't find specific venues for {query} in {location}.
            Here are some suggestions to help refine your search:
            1. Try a different location or expand your search radius
            2. Consider alternative venue types
            3. Adjust your capacity or budget requirements
            4. Be more specific about the type of venue
            
            Would you like to try any of these suggestions?"""

        # Get Gemini response
        response = get_gemini_response(f"Find {query} in {location}")
        
        # If Gemini fails, use template response
        if not response or "I'm having trouble" in response:
            response = f"I found {len(venues)} venues for {query} in {location}:\n\n"
            
            for i, venue in enumerate(venues, 1):
                response += f"{i}. {venue['name']}\n"
                response += f"   - Type: {venue.get('categories', [{}])[0].get('name', 'Venue')}\n"
                if venue.get('rating'):
                    response += f"   - Rating: {venue['rating']}/10\n"
                if venue.get('price'):
                    response += f"   - Price Level: {'$' * venue['price']}\n"
                if venue.get('location', {}).get('formatted_address'):
                    response += f"   - Address: {venue['location']['formatted_address']}\n"
                response += "\n"

            if len(venues) < 10:
                response += f"\nNote: I found {len(venues)} venues matching your criteria. Would you like to try a different search with broader parameters?"
            else:
                response += "\nWould you like more information about any of these venues or would you like to refine your search?"

        return response

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I found some venues for you, but I'm having trouble formatting the details. Would you like to try a different search?"

@app.get("/")
async def read_root():
    """Serve the main HTML file."""
    try:
        return FileResponse("templates/index.html")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving the application")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "healthy"})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Validate API keys
        if not FOURSQUARE_API_KEY or not GEMINI_API_KEY:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Missing API keys",
                    "response": "I'm currently having trouble accessing the venue database. Please try again in a few moments."
                }
            )

        # Validate message
        if not request.message or len(request.message.strip()) < 3:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "response": "Please provide more details about the venue you're looking for. For example: 'Find a wedding venue in New York for 200 people'"
                }
            )

        # Extract location and query from message
        message = request.message.lower()
        location = None
        query = None

        # Try to extract location
        location_pattern = r'(in|at|near|around|close to|within)\s+([^,.!?]+)'
        location_match = re.search(location_pattern, message)
        if location_match:
            location = location_match.group(2).strip()

        if not location:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "response": "I need to know where you're looking for venues. Please include a location in your request. For example: 'Find venues in New York'"
                }
            )

        # Get location coordinates
        lat, lng = get_geocode(location)
        if not lat or not lng:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "response": f"I couldn't find the location '{location}'. Please try being more specific or check the spelling. You can try:\n1. Using the full city name\n2. Adding the state or country\n3. Using a nearby landmark"
                }
            )

        # Extract query from message
        query = "venue"  # Default query
        if "wedding" in message:
            query = "wedding venue"
        elif "business" in message or "conference" in message:
            query = "conference center"
        elif "sports" in message:
            query = "sports venue"
        elif "restaurant" in message or "dining" in message:
            query = "restaurant"

        # Search for venues
        venues = search_foursquare_venues(
            location,
            query
        )
        
        # Get Gemini response
        response = generate_helpful_response(
            query,
            location,
            None,
            None,
            venues
        )
        
        return JSONResponse(
            content={
                "success": True,
                "response": response,
                "venues": venues
            }
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "response": "I'm having trouble processing your request right now. Please try again with a different query or try again in a few moments."
            }
        )

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    """Handle 404 Not Found errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found. Please check the URL and try again."}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: HTTPException):
    """Handle 500 Internal Server errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
