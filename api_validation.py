import os



import requests
def validate_groq_api_key(api_key: str) -> bool:
    """Return True if Groq API key is valid, else False."""
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def validate_openrouter_api_key(api_key: str) -> bool:
    """Validate OpenRouter API key. Returns True if valid, False otherwise."""
    try:
        response = requests.get(
            url="https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        data = response.json()
        
        # If response contains "data", key is valid
        if "data" in data:
            return True
        else:
            return False
    except Exception:
        return False