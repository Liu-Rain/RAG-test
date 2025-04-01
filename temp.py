import requests
import json
response = requests.get(
  url="https://openrouter.ai/api/v1/auth/key",
  headers={
    "Authorization": f"Bearer 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"
  }
)
print(json.dumps(response.json(), indent=2))