from langchain_core.tools import tool
from geopy import Nominatim
import requests

def get_location_coordinates(location: str) -> str:
    """Get the coordinates for a given location."""
    try:
        geolocator = Nominatim(user_agent="exercise_app")
        loc = geolocator.geocode(location)
        if loc:
            coordinates_dict = {
                "latitude": loc.latitude, 
                "longitude": loc.longitude
                }
    except Exception as e:
        print(f"Error fetching coordinates for {location}: {e}")
        coordinates_dict = {
            "latitude": 40.7128,  # Default to New York City coordinates
            "longitude": -74.0060
        }
    return coordinates_dict  # Coordinates for New York City

@tool
def get_weather(location: str, start_date: str, end_date: str) -> str:
    """Tool to get weather information for a given location and date range.
    Args:
        location (str): The location to get the weather for.
        start_date (str): The start date in YYYY-MM-DD format.
        end_date (str): The end date in YYYY-MM-DD format.
    
    Returns:
        dict: A dictionary containing weather information.
    """
    coordinates = get_location_coordinates(location)

    latitude = coordinates["latitude"]
    longitude = coordinates["longitude"]

    # Costruzione URL API
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation_probability,precipitation,wind_speed_10m,relative_humidity_2m"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Errore durante la richiesta API: {e}"
    
    print("Weather Tool - Weather data:")
    print(data)
    print(type(data))
    return data
