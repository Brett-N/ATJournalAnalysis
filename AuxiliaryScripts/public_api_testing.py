import json
import requests


def test_single_keys(key: str):
    # get the datetime for sept 5 2023 as a baseline stamp
    dt = 1695410451

    correctPackage = {
        "lat": 10,
        "lon": 10,
        "timezone": "Africa/Lagos",
        "timezone_offset": 3600,
        "data": [
            {
                "dt": 1695410451,
                "sunrise": 1695359354,
                "sunset": 1695402997,
                "temp": 81.79,
                "feels_like": 84.13,
                "pressure": 1009,
                "humidity": 60,
                "dew_point": 66.56,
                "uvi": 0,
                "clouds": 100,
                "visibility": 10000,
                "wind_speed": 6.89,
                "wind_deg": 136,
                "wind_gust": 15.43,
                "weather": [
                    {
                        "id": 804,
                        "main": "Clouds",
                        "description": "overcast clouds",
                        "icon": "04n",
                    }
                ],
            }
        ],
    }

    try:
        # get the weather data for the destination and date
        package = requests.get(
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat=10&lon=10&dt={dt}&appid={key}&units=imperial"
        )

    except:
        print(f"Error: api call failed on key {key}")
        print(package.json())
        return

    if package.status_code != 200:
        print(f"Error: api call failed on key {key}")
        print(package.json())
        return

    if "timezone" not in package.json().keys():
        print(f"Error: api call failed on key {key}")
        print(package.json())
        return

    print(json.dumps(package.json(), indent=4, sort_keys=True))
    print(f"the key {key} is valid! :)")


key = str(input("enter your key: "))

test_single_keys(key)
