import os
import json
import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

class ChatResponse(BaseModel):
    response: str
    venues: Optional[List[dict]] = None
    error: Optional[str] = None

def get_geocode(location: str) -> tuple:
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

def get_foursquare_venues(lat: float, lng: float, event_type: str, radius: int = 5000) -> List[dict]:
    """Get venues from Foursquare API based on location and event type."""
    try:
        # Map event types to Foursquare categories
        category_mapping = {
            'business': ['13035', '13036', '13037', '13038'],  # Business Centers, Conference Centers, Meeting Rooms, Co-working Spaces
            'sports': ['18008', '18009', '18010', '18011'],    # Sports Complexes, Stadiums, Arenas, Sports Clubs
            'wedding': ['13065', '13066', '13067', '13068'],   # Wedding Venues, Banquet Halls, Event Spaces, Reception Halls
            'social': ['13065', '13066', '13067', '13068'],    # Event Spaces, Banquet Halls, Party Venues, Reception Halls
            'graduation': ['13065', '13066', '13067', '13068'], # Event Spaces, Banquet Halls, Auditoriums, Reception Halls
            'exhibition': ['10000', '10001', '10002', '10003'], # Art Galleries, Museums, Exhibition Centers, Cultural Centers
            'dining': ['13065', '13066', '13067', '13068'],    # Restaurants, Cafes, Food Courts, Dining Halls
            'accommodation': ['19000', '19001', '19002'],      # Hotels, Resorts, Hostels
            'entertainment': ['10000', '10001', '10002', '10003'], # Theaters, Cinemas, Performance Venues
            'general': ['13065', '13066', '13067', '13068']    # General Event Spaces
        }

        categories = category_mapping.get(event_type, ['13065'])  # Default to event spaces
        
        headers = {
            "Authorization": FOURSQUARE_API_KEY,
            "accept": "application/json"
        }
        
        params = {
            "ll": f"{lat},{lng}",
            "radius": radius,
            "categories": ",".join(categories),
            "limit": 10,
            "sort": "DISTANCE"
        }
        
        response = requests.get(FOURSQUARE_BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        venues = []
        for place in data.get('results', []):
            venue = {
                'name': place.get('name', ''),
                'type': place.get('categories', [{}])[0].get('name', ''),
                'address': place.get('location', {}).get('formatted_address', ''),
                'rating': place.get('rating', 0),
                'price': place.get('price', 0),
                'capacity': place.get('capacity', 0),
                'image': place.get('photos', [{}])[0].get('prefix', '') + 'original' + place.get('photos', [{}])[0].get('suffix', ''),
                'location': {
                    'lat': place.get('geocodes', {}).get('main', {}).get('latitude'),
                    'lng': place.get('geocodes', {}).get('main', {}).get('longitude')
                }
            }
            venues.append(venue)
        
        return venues
    except Exception as e:
        logger.error(f"Foursquare API error: {str(e)}")
        return []

@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        context = request.context or {}
        event_type = context.get('event_type', 'general')
        location = context.get('location')
        capacity = context.get('capacity')
        budget = context.get('budget')

        # Get location coordinates
        lat, lng = None, None
        if location:
            lat, lng = get_geocode(location)
            if not lat or not lng:
                return ChatResponse(
                    response="I couldn't find that location. Please try being more specific about the location.",
                    venues=None
                )

        # Get venues from Foursquare
        venues = []
        if lat and lng:
            venues = get_foursquare_venues(lat, lng, event_type)
            
            # Filter venues based on capacity and budget if provided
            if capacity:
                venues = [v for v in venues if v.get('capacity', 0) >= capacity]
            if budget:
                venues = [v for v in venues if v.get('price', 0) <= budget]

        # Generate response using Gemini
        prompt = f"""
        You are a helpful venue finding assistant. A user is looking for venues for a {event_type} event.
        {f'Location: {location}' if location else ''}
        {f'Capacity needed: {capacity} people' if capacity else ''}
        {f'Budget: ${budget}' if budget else ''}
        
        Here are some relevant venues I found:
        {json.dumps(venues, indent=2) if venues else 'No venues found matching the criteria.'}
        
        Please provide a helpful response that:
        1. Acknowledges the user's request and the type of event they're looking for
        2. Mentions any specific requirements (location, capacity, budget)
        3. If venues are found:
           - List the top 3-5 venues with their key features
           - Mention ratings, capacity, and price if available
           - Highlight any unique features or amenities
        4. If no venues are found:
           - Suggest alternative locations or event types
           - Provide tips for refining the search
           - Ask for more specific requirements
        5. Keep the response friendly and conversational
        6. End with a question to help refine the search if needed
        """

        response = model.generate_content(prompt)
        
        return ChatResponse(
            response=response.text,
            venues=venues if venues else None
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
