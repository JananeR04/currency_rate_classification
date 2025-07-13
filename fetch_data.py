# fetch_data.py

import requests

def get_live_rate(base="USD", target="INR"):
    url = f"https://v6.exchangerate-api.com/v6/bcd963099e4c0afc6ef436ac/latest/{base}"
    response = requests.get(url)
    data = response.json()
    return data["conversion_rates"][target]
