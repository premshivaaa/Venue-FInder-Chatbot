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
model = genai.GenerativeModel('gemini-2.0-flash')

# Foursquare API configuration
FOURSQUARE_API_KEY = os.getenv('FOURSQUARE_API_KEY')
FOURSQUARE_BASE_URL = "https://api.foursquare.com/v3/places/search"
FOURSQUARE_PLACE_DETAILS_URL = "https://api.foursquare.com/v3/places/"
FOURSQUARE_PHOTOS_URL = "https://api.foursquare.com/v3/places/{}/photos"

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
    relevance_score: Optional[float]

class ChatResponse(BaseModel):
    response: str
    venues: Optional[List[dict]] = None
    error: Optional[str] = None

def clean_gemini_response(text: str) -> str:
    """Clean Gemini response by removing markdown formatting."""
    # Remove markdown bold syntax
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove other markdown syntax
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    return text.strip()

def get_venue_photos(fsq_id: str) -> List[str]:
    """Get photos for a venue from Foursquare."""
    try:
        headers = {
            "Authorization": FOURSQUARE_API_KEY,
            "accept": "application/json"
        }
        
        response = requests.get(
            FOURSQUARE_PHOTOS_URL.format(fsq_id),
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
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
        headers = {
            "Authorization": FOURSQUARE_API_KEY,
            "accept": "application/json"
        }
        
        response = requests.get(
            f"{FOURSQUARE_PLACE_DETAILS_URL}{fsq_id}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
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

    # Use Gemini to analyze the message
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
        
        Format: {{"event_type": "...", "location": "...", "capacity": number, "budget": number, "specific_requirements": [...], "keywords": [...]}}"""
        
        response = model.generate_content(prompt)
        try:
            gemini_data = json.loads(response.text)
            details.update(gemini_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse Gemini response")
    except Exception as e:
        logger.error(f"Error using Gemini for analysis: {str(e)}")

    # Fallback to regex if Gemini fails
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

    return details

def get_foursquare_venues(lat: float, lng: float, event_type: str, keywords: List[str], radius: int = 5000) -> List[dict]:
    """Get venues from Foursquare API based on location and event type."""
    try:
        category_mapping = {
            'business': {
                'categories': ['13035', '13036', '13037', '13038'],
                'description': 'business meeting or conference',
                'keywords': ['conference', 'meeting', 'business', 'corporate', 'office', 'work'],
                'exclude_categories': ['13065', '13066', '13067', '13068']
            },
            'sports': {
                'categories': ['18008', '18009', '18010', '18011', '18012', '18013', '18014'],
                'description': 'sports event or tournament',
                'keywords': ['sports', 'game', 'tournament', 'match', 'basketball', 'court', 'arena', 'stadium', 'gym', 'field', 'sport', 'athletic', 'fitness', 'training', 'competition'],
                'exclude_categories': ['13065', '13066', '13067', '13068']
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
            {"categories": ",".join(categories), "sort": "RATING"},
            {"query": " ".join(keywords), "sort": "RATING"},
            {"categories": ",".join(categories), "sort": "DISTANCE"},
            {"query": " ".join(keywords), "sort": "DISTANCE"}
        ]
        
        all_venues = []
        for strategy in search_strategies:
            params = {
                "ll": f"{lat},{lng}",
                "radius": radius,
                "limit": 20,
                **strategy
            }
            
            try:
                response = requests.get(FOURSQUARE_BASE_URL, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                for place in data.get('results', []):
                    # Skip venues that don't match the event type
                    venue_categories = [cat.get('id', '') for cat in place.get('categories', [])]
                    if any(cat in event_info.get('exclude_categories', []) for cat in venue_categories):
                        continue

                    # Get detailed venue information
                    details = get_venue_details(place.get('fsq_id', ''))
                    
                    # Get the best available image
                    image = None
                    if details.get('photos'):
                        image = details['photos'][0]
                    elif place.get('photos', [{}])[0].get('prefix'):
                        image = f"{place['photos'][0]['prefix']}original{place['photos'][0]['suffix']}"
                    
                    venue = {
                        'name': place.get('name', ''),
                        'type': place.get('categories', [{}])[0].get('name', ''),
                        'address': place.get('location', {}).get('formatted_address', ''),
                        'rating': place.get('rating', 0),
                        'price': place.get('price', 0),
                        'capacity': details.get('capacity', 0),
                        'image': image,
                        'location': {
                            'lat': place.get('geocodes', {}).get('main', {}).get('latitude'),
                            'lng': place.get('geocodes', {}).get('main', {}).get('longitude')
                        },
                        'description': details.get('description', ''),
                        'amenities': details.get('amenities', []),
                        'hours': details.get('hours', {}),
                        'contact': details.get('contact', {})
                    }
                    
                    # Calculate relevance score
                    venue['relevance_score'] = calculate_relevance_score(venue, event_type, keywords)
                    
                    # Add venue if not already in the list
                    if not any(v['name'] == venue['name'] for v in all_venues):
                        all_venues.append(venue)
                
            except Exception as e:
                logger.error(f"Error in search strategy: {str(e)}")
                continue
        
        # Sort by relevance score and rating
        all_venues.sort(key=lambda x: (x.get('relevance_score', 0), x.get('rating', 0)), reverse=True)
        
        # Filter out venues with very low relevance
        relevant_venues = [v for v in all_venues if v.get('relevance_score', 0) > 0.3]
        
        return relevant_venues[:10] if len(relevant_venues) > 10 else relevant_venues

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

        # Use Gemini to generate a natural response
        try:
            prompt = f"""You are a helpful venue finding assistant. A user is looking for venues for a {event_type} event in {location}.
            {f'Capacity needed: {capacity} people' if capacity else ''}
            {f'Budget: ${budget}' if budget else ''}
            
            Here are the venues I found:
            {json.dumps(venues, indent=2)}
            
            Please provide a helpful, conversational response that:
            1. Acknowledges the user's request
            2. Lists the top venues with their key features
            3. Mentions any specific requirements that were met
            4. Keeps the response friendly and natural
            5. Ends with a question to help refine the search if needed"""
            
            response = model.generate_content(prompt)
            return clean_gemini_response(response.text)
        except Exception as e:
            logger.error(f"Error using Gemini for response: {str(e)}")
            # Fallback to template response
            response = f"I found {len(venues)} venues for your {event_type} event in {location}:\n\n"
            
            for i, venue in enumerate(venues, 1):
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

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Extract event details from the message
        event_details = extract_event_details(request.message)
        
        # Validate location
        if not event_details['location']:
            return ChatResponse(
                response="Please specify a location for your event. For example: 'Find venues in New York'",
                venues=None
            )

        # Get location coordinates
        lat, lng = get_geocode(event_details['location'])
        if not lat or not lng:
            return ChatResponse(
                response=f"I couldn't find the location '{event_details['location']}'. Please try being more specific or check the spelling.",
                venues=None
            )

        # Get venues from Foursquare
        venues = get_foursquare_venues(
            lat, 
            lng, 
            event_details['event_type'],
            event_details['keywords']
        )
        
        # Filter venues based on capacity and budget if provided
        if event_details['capacity']:
            venues = [v for v in venues if v.get('capacity', 0) >= event_details['capacity']]
        if event_details['budget']:
            venues = [v for v in venues if v.get('price', 0) <= event_details['budget']]

        # Generate response
        response = generate_helpful_response(
            event_details['event_type'],
            event_details['location'],
            event_details['capacity'],
            event_details['budget'],
            venues
        )
        
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
