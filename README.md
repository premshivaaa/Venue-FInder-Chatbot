# VenueFinder - AI-Powered Venue Discovery Assistant

VenueFinder is an intelligent chatbot that helps users discover venues for various purposes, from sports facilities to restaurants and meeting places. It provides real-time information, ratings, and directions to venues worldwide.

## Features

- 🤖 AI-powered conversational interface
- 🏟️ Sports venue discovery with capacity information
- 🍽️ Restaurant and personal venue recommendations
- 📍 Real-time location-based search
- ⭐ Venue ratings and reviews
- 🗺️ Integrated Google Maps directions
- 💾 Chat history and venue bookmarking
- 🌓 Dark/Light theme toggle
- 📱 Responsive design for all devices

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python with FastAPI
- AI: Google's Gemini Pro API
- Location Services: MapTiler API
- Venue Data: Foursquare API

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/venue-finder.git
cd venue-finder
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key
FOURSQUARE_API_KEY=your_foursquare_api_key
MAPTILER_API_KEY=your_maptiler_api_key
```

5. Run the application:
```bash
uvicorn api.chat:app --reload
```

6. Open your browser and navigate to `http://localhost:8000`

## API Keys Setup

1. **Gemini API Key**: 
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Foursquare API Key**:
   - Visit [Foursquare Developer Portal](https://developer.foursquare.com/)
   - Create an account and get your API key

3. **MapTiler API Key**:
   - Visit [MapTiler Cloud](https://cloud.maptiler.com/auth/signup/)
   - Create an account and get your API key

## Usage

1. Type your venue-related query in the chat input
2. The AI will understand your request and search for relevant venues
3. View venue details, including:
   - Photos
   - Ratings
   - Price levels
   - Capacity (for sports venues)
   - Exact location
4. Click "Get Directions" to open Google Maps navigation
5. Save interesting venues for later reference
6. View your chat history and saved venues

## Deployment

This application is ready to be deployed on Vercel:

1. Fork this repository
2. Connect your GitHub account to Vercel
3. Create a new project and import the repository
4. Add your environment variables in the Vercel project settings
5. Deploy!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 