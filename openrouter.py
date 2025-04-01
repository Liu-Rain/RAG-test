import requests
import json


conversation_history = [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": f"Context: 'We are in Taiwan'"},
        ]

while True:

    try:
        user_input = input("You: ")
        conversation_history.append({"role": "user", "content": user_input})

        response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
        },

        data=json.dumps({
            "model": "google/gemini-2.5-pro-exp-03-25:free", # Optional
            "messages": conversation_history
        })
        )

        text = response.json()

        ai_response = text['choices'][0]['message']['content']
        print("AI:", ai_response)
        conversation_history.append({"role": "assistant", "content": ai_response})

    except:
        print(text.keys())
        print(text)
        Exception("Stop!")




