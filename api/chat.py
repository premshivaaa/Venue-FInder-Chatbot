import os
import json
import logging
import re
from typing import Optional, List, Dict, Tuple, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
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
import traceback

# Load environment variables
load_dotenv()

# Configure logging for Vercel
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log') if os.path.exists('/tmp') else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with Vercel-specific settings
app = FastAPI(
    title="Venue Finder API",
    description="API for finding and recommending venues",
    version="1.0.0",
    docs_url="/api/docs" if os.getenv('VERCEL_ENV') != 'production' else None,
    redoc_url="/api/redoc" if os.getenv('VERCEL_ENV') != 'production' else None
)

# Configure CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel-specific configuration
VERCEL_ENV = os.getenv('VERCEL_ENV', 'development')
IS_VERCEL = bool(os.getenv('VERCEL'))
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB

# Configure Gemini API
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')
    MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set in environment variables")
        raise RuntimeError("GEMINI_API_KEY is required")
    if not FOURSQUARE_API_KEY:
        logger.error("FOURSQUARE_API_KEY is not set in environment variables")
        raise RuntimeError("FOURSQUARE_API_KEY is required")
    if not MAPTILER_API_KEY:
        logger.error("MAPTILER_API_KEY is not set in environment variables")
        raise RuntimeError("MAPTILER_API_KEY is required")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Successfully configured Gemini API")

    # Test API connections
    try:
        # Test Foursquare API
        test_response = requests.get(
            FOURSQUARE_BASE_URL,
            headers=FOURSQUARE_HEADERS,
            params={'limit': 1},
            timeout=5
        )
        test_response.raise_for_status()
        logger.info("Successfully connected to Foursquare API")

        # Test MapTiler API
        test_response = requests.get(
            f"{MAPTILER_GEOCODING_URL}/test.json",
            params={'key': MAPTILER_API_KEY},
            headers=MAPTILER_HEADERS,
            timeout=5
        )
        test_response.raise_for_status()
        logger.info("Successfully connected to MapTiler API")
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection test failed: {str(e)}", exc_info=True)
        raise RuntimeError("Failed to connect to one or more APIs")

except Exception as e:
    logger.critical(f"Critical initialization error: {str(e)}", exc_info=True)
    raise RuntimeError("Failed to initialize the application")

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

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add timing and error handling."""
    start_time = time.time()
    try:
        # Check request size
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > MAX_REQUEST_SIZE:
                raise HTTPException(status_code=413, detail="Request too large")

        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(e),
                "traceback": traceback.format_exc() if VERCEL_ENV != 'production' else None
            }
        )

@app.get("/api/health")
async def health_check(background_tasks: BackgroundTasks):
    """Enhanced health check endpoint for Vercel."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": VERCEL_ENV,
        "is_vercel": IS_VERCEL,
        "api_status": {}
    }

    # Check API connections in background
    async def check_apis():
        try:
            # Test Foursquare API
            response = requests.get(
                FOURSQUARE_BASE_URL,
                headers=FOURSQUARE_HEADERS,
                params={'limit': 1},
                timeout=5
            )
            health_status["api_status"]["foursquare"] = response.status_code == 200
        except Exception as e:
            health_status["api_status"]["foursquare"] = False
            logger.error(f"Foursquare API health check failed: {str(e)}")

        try:
            # Test MapTiler API
            response = requests.get(
                f"{MAPTILER_GEOCODING_URL}/test.json",
                params={'key': MAPTILER_API_KEY},
                headers=MAPTILER_HEADERS,
                timeout=5
            )
            health_status["api_status"]["maptiler"] = response.status_code == 200
        except Exception as e:
            health_status["api_status"]["maptiler"] = False
            logger.error(f"MapTiler API health check failed: {str(e)}")

        try:
            # Test Gemini API
            if model:
                response = model.generate_content("test")
                health_status["api_status"]["gemini"] = True
            else:
                health_status["api_status"]["gemini"] = False
        except Exception as e:
            health_status["api_status"]["gemini"] = False
            logger.error(f"Gemini API health check failed: {str(e)}")

    background_tasks.add_task(check_apis)
    return health_status

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "traceback": traceback.format_exc() if VERCEL_ENV != 'production' else None
        }
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with improved error handling and validation."""
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")  # Log first 100 chars
        
        # Extract event details
        try:
            event_details = extract_event_details(request.message)
            logger.info(f"Extracted event details: {event_details}")
        except Exception as e:
            logger.error(f"Error extracting event details: {str(e)}", exc_info=True)
            return ChatResponse(
                response="I'm having trouble understanding your request. Could you please provide more details?",
                error="Event details extraction failed"
            )

        if not event_details['location']:
            logger.warning("No location specified in request")
            return ChatResponse(
                response="I need to know the location to help you find venues. Please specify where you're looking.",
                error="Location not specified"
            )

        # Search for venues
        try:
            venues = search_foursquare_venues(
                event_details['location'],
                f"{event_details['event_type']} venue",
                limit=5
            )
            logger.info(f"Found {len(venues)} venues for location: {event_details['location']}")
        except Exception as e:
            logger.error(f"Error searching venues: {str(e)}", exc_info=True)
            return ChatResponse(
                response=f"I encountered an error while searching for venues in {event_details['location']}. Please try again.",
                error="Venue search failed"
            )

        if not venues:
            logger.warning(f"No venues found for location: {event_details['location']}")
            return ChatResponse(
                response=f"I couldn't find any venues in {event_details['location']} for {event_details['event_type']}. Please try a different location or event type.",
                error="No venues found"
            )

        # Process venues
        try:
            processed_venues = process_venues(
                venues,
                event_details['event_type'],
                event_details['requirements']
            )
            logger.info(f"Processed {len(processed_venues)} venues")
        except Exception as e:
            logger.error(f"Error processing venues: {str(e)}", exc_info=True)
            return ChatResponse(
                response="I found some venues but had trouble processing them. Please try again.",
                error="Venue processing failed"
            )

        # Generate response
        try:
            response = get_gemini_response(
                f"Based on the following venues, provide a helpful response to the user's request: {request.message}\n\n"
                f"Available venues: {json.dumps(processed_venues[:3])}"
            )
            logger.info("Successfully generated Gemini response")
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}", exc_info=True)
            return ChatResponse(
                response="I found some venues but had trouble generating a response. Here are the venues I found:",
                venues=processed_venues[:3],
                error="Response generation failed"
            )

        return ChatResponse(
            response=response,
            venues=processed_venues[:3]
        )

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return ChatResponse(
            response="I'm sorry, but I couldn't process your request. Please check your input and try again.",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
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
