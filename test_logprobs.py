import requests
import json

URL = "http://10.70.13.33:11434/v1/chat/completions"
API_KEY = "sk-RZSBTkuZYOeXULKBTKupkA"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "model": "deepseek-32k",
    "messages": [
        {
            "role": "user",
            "content": "Return only 17*23."
        }
    ],
    "max_tokens": 8,
    "logprobs": True,
    "top_logprobs": 5,
    "extra_body": {
        "think": False
    }
}

response = requests.post(
    URL,
    headers=headers,
    json=payload,
    timeout=60
)

response.raise_for_status()

data = response.json()

print(json.dumps(data, indent=2))