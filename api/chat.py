import os
import json
import logging
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
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
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Foursquare API configuration
FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')
FOURSQUARE_BASE_URL = "https://api.foursquare.com/v3/places/search"
FOURSQUARE_PLACE_DETAILS_URL = "https://api.foursquare.com/v3/places/"

# MapTiler API configuration
MAPTILER_API_KEY = os.getenv('MAPTILER_API_KEY')
MAPTILER_GEOCODING_URL = "https://api.maptiler.com/geocoding"

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

class ChatResponse(BaseModel):
    response: str
    venues: Optional[List[dict]] = None
    error: Optional[str] = None

def get_geocode(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude and longitude for a location using MapTiler."""
    try:
        response = requests.get(
            f"{MAPTILER_GEOCODING_URL}/{location}.json",
            params={"key": MAPTILER_API_KEY}
        )
        response.raise_for_status()
        data = response.json()
        if data.get('features'):
            coords = data['features'][0]['center']
            return coords[1], coords[0]  # latitude, longitude
        return None, None
    except Exception as e:
        logger.error(f"Geocoding error: {str(e)}")
        return None, None

def get_venue_details(fsq_id: str) -> Dict:
    """Get detailed information about a venue from Foursquare."""
    try:
        headers = {
            "Authorization": FOURSQUARE_API_KEY,
            "accept": "application/json"
        }
        
        response = requests.get(
            f"{FOURSQUARE_PLACE_DETAILS_URL}{fsq_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error getting venue details: {str(e)}")
        return {}

def get_foursquare_venues(lat: float, lng: float, event_type: str, radius: int = 5000) -> List[dict]:
    """Get venues from Foursquare API based on location and event type."""
    try:
        # Map event types to Foursquare categories with detailed descriptions
        category_mapping = {
            'business': {
                'categories': ['13035', '13036', '13037', '13038'],
                'description': 'business meeting or conference',
                'keywords': ['conference', 'meeting', 'business', 'corporate']
            },
            'sports': {
                'categories': ['18008', '18009', '18010', '18011'],
                'description': 'sports event or tournament',
                'keywords': ['sports', 'game', 'tournament', 'match']
            },
            'wedding': {
                'categories': ['13065', '13066', '13067', '13068'],
                'description': 'wedding or reception',
                'keywords': ['wedding', 'reception', 'marriage', 'ceremony']
            },
            'social': {
                'categories': ['13065', '13066', '13067', '13068'],
                'description': 'social event or party',
                'keywords': ['party', 'celebration', 'social', 'event']
            },
            'graduation': {
                'categories': ['13065', '13066', '13067', '13068'],
                'description': 'graduation ceremony',
                'keywords': ['graduation', 'ceremony', 'commencement']
            },
            'exhibition': {
                'categories': ['10000', '10001', '10002', '10003'],
                'description': 'art exhibition or gallery',
                'keywords': ['exhibition', 'gallery', 'art', 'show']
            },
            'dining': {
                'categories': ['13065', '13066', '13067', '13068'],
                'description': 'restaurant or dining venue',
                'keywords': ['restaurant', 'cafe', 'dining', 'food']
            },
            'accommodation': {
                'categories': ['19000', '19001', '19002'],
                'description': 'hotel or accommodation',
                'keywords': ['hotel', 'resort', 'accommodation', 'stay']
            },
            'entertainment': {
                'categories': ['10000', '10001', '10002', '10003'],
                'description': 'entertainment venue',
                'keywords': ['theater', 'cinema', 'movie', 'play']
            },
            'general': {
                'categories': ['13065', '13066', '13067', '13068'],
                'description': 'general event space',
                'keywords': ['venue', 'space', 'location', 'place']
            }
        }

        event_info = category_mapping.get(event_type, category_mapping['general'])
        categories = event_info['categories']
        
        headers = {
            "Authorization": FOURSQUARE_API_KEY,
            "accept": "application/json"
        }
        
        # Try different search strategies
        search_strategies = [
            {"categories": ",".join(categories), "sort": "DISTANCE"},
            {"categories": ",".join(categories), "sort": "RATING"},
            {"query": " ".join(event_info['keywords']), "sort": "DISTANCE"},
            {"query": " ".join(event_info['keywords']), "sort": "RATING"}
        ]
        
        all_venues = []
        for strategy in search_strategies:
            params = {
                "ll": f"{lat},{lng}",
                "radius": radius,
                "limit": 10,
                **strategy
            }
            
            try:
                response = requests.get(FOURSQUARE_BASE_URL, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                for place in data.get('results', []):
                    # Get detailed venue information
                    details = get_venue_details(place.get('fsq_id', ''))
                    
                    venue = {
                        'name': place.get('name', ''),
                        'type': place.get('categories', [{}])[0].get('name', ''),
                        'address': place.get('location', {}).get('formatted_address', ''),
                        'rating': place.get('rating', 0),
                        'price': place.get('price', 0),
                        'capacity': details.get('capacity', 0),
                        'image': place.get('photos', [{}])[0].get('prefix', '') + 'original' + place.get('photos', [{}])[0].get('suffix', ''),
                        'location': {
                            'lat': place.get('geocodes', {}).get('main', {}).get('latitude'),
                            'lng': place.get('geocodes', {}).get('main', {}).get('longitude')
                        },
                        'description': details.get('description', ''),
                        'amenities': details.get('amenities', []),
                        'hours': details.get('hours', {}),
                        'contact': details.get('contact', {})
                    }
                    
                    # Add venue if not already in the list
                    if not any(v['name'] == venue['name'] for v in all_venues):
                        all_venues.append(venue)
                
            except Exception as e:
                logger.error(f"Error in search strategy: {str(e)}")
                continue
        
        return all_venues
    except Exception as e:
        logger.error(f"Foursquare API error: {str(e)}")
        return []

def generate_helpful_response(event_type: str, location: str, capacity: int, budget: int, venues: List[dict]) -> str:
    """Generate a helpful response based on the search results."""
    try:
        if not venues:
            return f"""I couldn't find specific venues for your {event_type} event in {location}.
            Here are some suggestions to help refine your search:
            1. Try a different location or expand your search radius
            2. Consider alternative venue types
            3. Adjust your capacity or budget requirements
            4. Be more specific about the type of event
            
            Would you like to try any of these suggestions?"""

        # Sort venues by rating and relevance
        sorted_venues = sorted(venues, key=lambda x: (x.get('rating', 0), x.get('capacity', 0)), reverse=True)
        top_venues = sorted_venues[:5]

        response = f"I found some great venues for your {event_type} event in {location}:\n\n"
        
        for i, venue in enumerate(top_venues, 1):
            response += f"{i}. {venue['name']}\n"
            response += f"   - Type: {venue['type']}\n"
            if venue.get('rating'):
                response += f"   - Rating: {venue['rating']}/10\n"
            if venue.get('capacity'):
                response += f"   - Capacity: {venue['capacity']} people\n"
            if venue.get('price'):
                response += f"   - Price Level: {'$' * venue['price']}\n"
            if venue.get('description'):
                response += f"   - Description: {venue['description'][:100]}...\n"
            response += "\n"

        response += "Would you like more information about any of these venues or would you like to refine your search?"
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Validate API keys
        if not FOURSQUARE_API_KEY or not MAPTILER_API_KEY:
            return ChatResponse(
                response="Service configuration error. Please check the API keys.",
                error="Missing API keys"
            )

        context = request.context or {}
        event_type = context.get('event_type', 'general')
        location = context.get('location')
        capacity = context.get('capacity')
        budget = context.get('budget')

        # Validate input
        if not location:
            return ChatResponse(
                response="Please specify a location for your event. For example: 'Find venues in New York'",
                venues=None
            )

        # Get location coordinates
        lat, lng = get_geocode(location)
        if not lat or not lng:
            return ChatResponse(
                response=f"I couldn't find the location '{location}'. Please try being more specific or check the spelling.",
                venues=None
            )

        # Get venues from Foursquare
        venues = get_foursquare_venues(lat, lng, event_type)
        
        # Filter venues based on capacity and budget if provided
        if capacity:
            venues = [v for v in venues if v.get('capacity', 0) >= capacity]
        if budget:
            venues = [v for v in venues if v.get('price', 0) <= budget]

        # Generate response
        response = generate_helpful_response(event_type, location, capacity, budget, venues)
        
        return ChatResponse(
            response=response,
            venues=venues if venues else None
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return ChatResponse(
            response="I encountered an error while processing your request. Please try again with a different query.",
            error=str(e)
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
