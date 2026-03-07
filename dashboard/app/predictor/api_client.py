import requests

def api_predict(api_url: str, payload: dict) -> dict:
    """
    Para futuro: consume API real.
    Espera endpoint POST {api_url}/api/v1/predict
    """
    url = f"{api_url}/api/v1/predict"
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()
