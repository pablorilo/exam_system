from openai import OpenAI

def main():
    client = OpenAI(
        api_key="sk-0088aa4da32d40a0aca0807067367ac5",
        base_url="https://api.deepseek.com"  # 👈 importante
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Eres un asistente que responde solo basándose en el texto proporcionado."},
            {"role": "user", "content": "¿Cuál es la capital de Francia?"}
        ]
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
