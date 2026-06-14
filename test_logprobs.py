import requests
import json

URL = "http://10.70.13.33:11434/v1/chat/completions"
API_KEY = "sk-RZSBTkuZYOeXULKBTKupkA"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "model": "gemma4:31b",
    "messages": [
        {
            "role": "user",
            "content": "Answer with either Yes or No only. Is red a color?"
        }
    ],
    "temperature": 0,
    "max_tokens": 10,

    # Request token probabilities
    "logprobs": True,
    "top_logprobs": 10,

    "chat_template_kwargs": {
        "enable_thinking": False
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

print("=" * 80)
print("CHOICE KEYS")
print(data["choices"][0].keys())

print("\n" + "=" * 80)
print("FULL CHOICE")
print(json.dumps(data["choices"][0], indent=2))

print("\n" + "=" * 80)
print("MESSAGE")
print(data["choices"][0]["message"]["content"])

if "logprobs" in data["choices"][0]:
    print("\n" + "=" * 80)
    print("LOGPROBS")
    print(json.dumps(data["choices"][0]["logprobs"], indent=2))
else:
    print("\nNo logprobs returned.")