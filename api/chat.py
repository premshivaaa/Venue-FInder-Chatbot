import os
import json
import logging
import re
from typing import Optional, List, Dict, Tuple, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import google.generativeai as genai
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
import urllib.parse
import time
from functools import lru_cache
from ratelimit import limits, sleep_and_retry

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

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')
MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')

# Validate API keys
if not all([GEMINI_API_KEY, FOURSQUARE_API_KEY, MAPTILER_API_KEY]):
    logger.error("Missing required API keys in environment variables")
    raise RuntimeError("Missing required API keys")

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    model = None

# API configuration
FOURSQUARE_BASE_URL = "https://api.foursquare.com/v3/places/search"
FOURSQUARE_PLACE_DETAILS_URL = "https://api.foursquare.com/v3/places/"
FOURSQUARE_PHOTOS_URL = "https://api.foursquare.com/v3/places/{}/photos"
MAPTILER_GEOCODING_URL = "https://api.maptiler.com/geocoding"

# API Headers
FOURSQUARE_HEADERS = {
    "Authorization": FOURSQUARE_API_KEY,
    "accept": "application/json"
}

MAPTILER_HEADERS = {
    "accept": "application/json"
}

# Rate limiting configuration
ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 30

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def rate_limited_request(url: str, headers: dict, params: dict = None, timeout: int = 10) -> requests.Response:
    """Make a rate-limited HTTP request."""
    return requests.get(url, headers=headers, params=params, timeout=timeout)

# Cache configuration
@lru_cache(maxsize=1000)
def cached_geocode(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Cached version of get_geocode."""
    return get_geocode(location)

@lru_cache(maxsize=1000)
def cached_venue_details(fsq_id: str) -> Dict[str, Any]:
    """Cached version of get_venue_details."""
    return get_venue_details(fsq_id)

class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None

    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message is too long (max 1000 characters)')
        return v.strip()

    @validator('context')
    def validate_context(cls, v):
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError('Context must be a dictionary')
            if len(v) > 10:
                raise ValueError('Context has too many items (max 10)')
        return v

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
                headers=MAPTILER_HEADERS,
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(geocode_request)
        
        if data.get('features'):
            coords = data['features'][0]['center']
            return coords[1], coords[0]  # latitude, longitude
        return None, None
    except Exception as e:
        logger.error(f"Geocoding error for location '{location}': {str(e)}")
        return None, None

def get_venue_photos(fsq_id: str) -> List[str]:
    """Get photos for a venue from Foursquare."""
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
        
        photos = []
        for photo in data.get('items', []):
            if 'prefix' in photo and 'suffix' in photo:
                photos.append(f"{photo['prefix']}original{photo['suffix']}")
        
        return photos[:3]  # Return max 3 photos
    except Exception as e:
        logger.error(f"Error getting venue photos for {fsq_id}: {str(e)}")
        return []

def get_venue_details(fsq_id: str) -> Dict[str, Any]:
    """Get detailed information about a venue from Foursquare."""
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
    
    for venue in raw_venues:
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
    
    # Sort by relevance score descending
    return sorted(processed_venues, key=lambda x: x.get('relevance_score', 0), reverse=True)

def calculate_relevance_score(venue: dict, event_type: str, keywords: List[str]) -> float:
    """Calculate a relevance score for a venue based on the search criteria."""
    score = 0.0
    
    # Base score from rating
    score += venue.get('rating', 0) * 0.2
    
    # Category match
    venue_categories = [cat.get('name', '').lower() for cat in venue.get('categories', [])]
    for keyword in keywords:
        if any(keyword.lower() in cat for cat in venue_categories):
            score += 0.3
    
    # Description match
    description = venue.get('description', '').lower()
    for keyword in keywords:
        if keyword.lower() in description:
            score += 0.2
    
    # Event type match
    if event_type and event_type.lower() in description:
        score += 0.3
    
    # Capacity consideration
    capacity = venue.get('capacity', 0)
    if capacity > 0:
        score += min(capacity / 1000, 0.2)  # Normalize capacity score
    
    return min(score, 1.0)  # Cap score at 1.0

def extract_event_details(message: str) -> Dict[str, Any]:
    """Extract event details from user message using Gemini."""
    try:
        prompt = f"""
        Extract the following information from this message about an event:
        - Event type (e.g., wedding, conference, party)
        - Location (city, state, or specific address)
        - Number of attendees (if mentioned)
        - Date (if mentioned)
        - Any specific requirements or preferences
        
        Message: {message}
        
        Return the information in JSON format with these keys:
        event_type, location, attendees, date, requirements
        """
        
        def gemini_request():
            response = model.generate_content(prompt)
            return response.text
        
        result = retry_with_backoff(gemini_request)
        
        try:
            data = json.loads(result)
            return {
                'event_type': data.get('event_type', ''),
                'location': data.get('location', ''),
                'attendees': data.get('attendees', 0),
                'date': data.get('date', ''),
                'requirements': data.get('requirements', [])
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini response: {result}")
            return {
                'event_type': '',
                'location': '',
                'attendees': 0,
                'date': '',
                'requirements': []
            }
    except Exception as e:
        logger.error(f"Error extracting event details: {str(e)}")
        return {
            'event_type': '',
            'location': '',
            'attendees': 0,
            'date': '',
            'requirements': []
        }

def search_foursquare_venues(location: str, query: str, limit: int = 10) -> List[dict]:
    """Search for venues using Foursquare API with rate limiting."""
    try:
        lat, lng = cached_geocode(location)
        if not lat or not lng:
            logger.error(f"Could not geocode location: {location}")
            return []

        params = {
            'll': f"{lat},{lng}",
            'query': query,
            'limit': limit,
            'radius': 50000,  # 50km radius
            'sort': 'DISTANCE'
        }

        def venues_request():
            response = rate_limited_request(
                FOURSQUARE_BASE_URL,
                headers=FOURSQUARE_HEADERS,
                params=params
            )
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(venues_request)
        return data.get('results', [])
    except Exception as e:
        logger.error(f"Error searching Foursquare venues: {str(e)}")
        return []

def get_gemini_response(message: str) -> str:
    """Get response from Gemini with improved error handling."""
    try:
        def gemini_request():
            response = model.generate_content(message)
            return response.text

        result = retry_with_backoff(gemini_request)
        return clean_gemini_response(result)
    except Exception as e:
        logger.error(f"Error getting Gemini response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

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
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with improved error handling and validation."""
    try:
        # Extract event details
        event_details = extract_event_details(request.message)
        if not event_details['location']:
            return ChatResponse(
                response="I need to know the location to help you find venues. Please specify where you're looking.",
                error="Location not specified"
            )

        # Search for venues
        venues = search_foursquare_venues(
            event_details['location'],
            f"{event_details['event_type']} venue",
            limit=5
        )

        if not venues:
            return ChatResponse(
                response=f"I couldn't find any venues in {event_details['location']} for {event_details['event_type']}. Please try a different location or event type.",
                error="No venues found"
            )

        # Process venues
        processed_venues = process_venues(
            venues,
            event_details['event_type'],
            event_details['requirements']
        )

        # Generate response
        response = get_gemini_response(
            f"Based on the following venues, provide a helpful response to the user's request: {request.message}\n\n"
            f"Available venues: {json.dumps(processed_venues[:3])}"
        )

        return ChatResponse(
            response=response,
            venues=processed_venues[:3]  # Return top 3 most relevant venues
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return ChatResponse(
            response="I'm sorry, but I couldn't process your request. Please check your input and try again.",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return ChatResponse(
            response="I'm sorry, but I encountered an error while processing your request. Please try again later.",
            error="Internal server error"
        )

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    """Handle 404 Not Found errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found."}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: HTTPException):
    """Handle 500 Internal Server errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
