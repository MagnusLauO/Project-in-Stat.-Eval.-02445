from openai import OpenAI

# API KEY: sk-0e1d62e63e10444b99b753f0991a2225
client = OpenAI(api_key="sk-0e1d62e63e10444b99b753f0991a2225", base_url="https://api.deepseek.com")

def generate_response(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=1.0
    )
    return response.choices[0].message.content

prompt = "Hello DeepSeek, I am a male and 25 years old. I work as a nurse in Denmark. Please respond with my expected salary in the CSV format: <gender>,<age>,<occupation>,<salary(range)>."
print(generate_response(prompt))
