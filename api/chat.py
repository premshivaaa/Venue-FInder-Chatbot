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

# Configure Gemini AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Foursquare API configuration
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
FOURSQUARE_API_URL = "https://api.foursquare.com/v3"

# MapTiler API configuration
MAPTILER_API_KEY = os.getenv("MAPTILER_API_KEY")

class ChatRequest(BaseModel):
    message: str

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
    venues: Optional[List[Venue]] = None

def get_venue_type(query: str) -> str:
    """Determine if the query is about sports venues or personal venues."""
    sports_keywords = ["sports", "stadium", "arena", "field", "court", "gym"]
    if any(keyword in query.lower() for keyword in sports_keywords):
        return "sports"
    return "personal"

async def get_coordinates(location: str) -> tuple:
    """Get coordinates for a location using MapTiler's geocoding API."""
    try:
        url = f"https://api.maptiler.com/geocoding/{location}.json?key={MAPTILER_API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if data["features"]:
            coordinates = data["features"][0]["geometry"]["coordinates"]
            return coordinates[1], coordinates[0]  # lat, lng
        return None, None
    except Exception as e:
        logger.error(f"Error getting coordinates: {e}")
        return None, None

async def search_venues(query: str, location: str) -> List[Venue]:
    """Search for venues using Foursquare API."""
    try:
        # Get coordinates for the location
        lat, lng = await get_coordinates(location)
        if not lat or not lng:
            raise HTTPException(status_code=400, detail="Location not found")

        # Prepare Foursquare API request
        headers = {
            "Accept": "application/json",
            "Authorization": FOURSQUARE_API_KEY
        }
        
        params = {
            "query": query,
            "ll": f"{lat},{lng}",
            "radius": 5000,
            "limit": 5,
            "sort": "RATING"
        }
        
        response = requests.get(
            f"{FOURSQUARE_API_URL}/places/search",
            headers=headers,
            params=params
        )
        
        venues_data = response.json()["results"]
        venues = []
        
        for venue in venues_data:
            # Get additional venue details
            venue_id = venue["fsq_id"]
            details_response = requests.get(
                f"{FOURSQUARE_API_URL}/places/{venue_id}",
                headers=headers
            )
            details = details_response.json()
            
            # Get venue photos
            photos_response = requests.get(
                f"{FOURSQUARE_API_URL}/places/{venue_id}/photos",
                headers=headers
            )
            photos = photos_response.json()
            
            # Create venue object
            venue_obj = Venue(
                name=venue.get("name", ""),
                type=venue.get("categories", [{}])[0].get("name", "Venue"),
                address=venue.get("location", {}).get("formatted_address", ""),
                rating=venue.get("rating", None),
                price=details.get("price", None),
                capacity=details.get("capacity", None),
                image=photos[0]["prefix"] + "original" + photos[0]["suffix"] if photos else None,
                latitude=venue["geocodes"]["main"]["latitude"],
                longitude=venue["geocodes"]["main"]["longitude"]
            )
            venues.append(venue_obj)
        
        return venues
    except Exception as e:
        logger.error(f"Error searching venues: {e}")
        raise HTTPException(status_code=500, detail="Error searching venues")

@app.get("/")
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/api/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    try:
        # Generate AI response
        chat = model.start_chat(history=[])
        response = chat.send_message(f"""
        You are a venue finder assistant. Help the user find venues based on their query:
        "{request.message}"
        
        If this is not a venue-related query, politely inform the user that you can only help with venue-related questions.
        If it is a venue query, extract the location and type of venue they're looking for.
        """)
        
        # Check if it's a venue-related query
        if "sorry" in response.text.lower() or "can only help" in response.text.lower():
            return ChatResponse(
                response="I apologize, but I can only help you find venues and locations. Please ask me about finding sports venues, restaurants, meeting places, or other locations!"
            )
        
        # Extract location and venue type from the query
        venue_type = get_venue_type(request.message)
        
        # Use AI to extract location
        location_response = chat.send_message(f"Extract only the location/city name from this query: {request.message}")
        location = location_response.text.strip()
        
        # Search for venues
        venues = await search_venues(request.message, location)
        
        # Generate a natural response
        venue_response = f"I found some {venue_type} venues in {location} that might interest you. "
        venue_response += "Here are the top recommendations based on ratings and reviews:"
        
        return ChatResponse(
            response=venue_response,
            venues=venues
        )
        
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 