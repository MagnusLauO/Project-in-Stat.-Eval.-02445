from openai import OpenAI
import random
import csv
import time

# Configuration
API_KEY = "sk-0e1d62e63e10444b99b753f0991a2226"
MODEL = "deepseek-chat"
OUTPUT_FILE = "salaries.csv"
OCCUPATION_MAP = {
    1: "software developer",
    2: "teacher",
    3: "nurse",
    4: "police officer",
    5: "marketing consultant"
}
GENDER_MAP = {'M': 'male', 'F': 'female'}

# Initialize client
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# Prompt template - keep this EXACT structure
PROMPT_TEMPLATE = """What is my expected pay in DKK before taxes as a {age} year old {gender} working as a {occupation} in Denmark. Provide the answer as comma separated values on the form: gender,age,occupation,lower_wage,upper_wage
- gender is either M (male) or F (female)
- age is discreet value
- occupation is made nominal in groups 1 to 5: 
    - 1: software developer
    - 2: teacher
    - 3: nurse
    - 4: police officer
    - 5: marketing consultant
- lower_wage is a continuous number
- upper_wage is a continuous number
Your response should be of low temperature in LLM-terms."""

def generate_response(prompt):
    """Get API response with error handling"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns strictly formatted data."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return None

# Generate samples
with open(OUTPUT_FILE, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['gender', 'age', 'occupation', 'lower_wage', 'upper_wage'])
    
    total_samples = 0
    
    for gender_code in ['M', 'F']:
        for occupation_id in range(1, 6):
            for _ in range(300):  # 300 samples per occupation per gender
                # Generate random age between 25-65
                age = random.randint(25, 65)
                
                # Build prompt
                prompt = PROMPT_TEMPLATE.format(
                    age=age,
                    gender=GENDER_MAP[gender_code],
                    occupation=OCCUPATION_MAP[occupation_id]
                )
                
                # Get API response with retries
                response = None
                for attempt in range(3):  # Max 3 attempts
                    response = generate_response(prompt)
                    if response and len(response.split(',')) == 5:
                        break
                    time.sleep(1)  # Wait before retry
                    print(f"Retrying {gender_code}/{occupation_id} (attempt {attempt+1})")
                
                # Process valid response
                if response and len(response.split(',')) == 5:
                    # Extract values from response
                    parts = response.split(',')
                    # Ensure occupation matches requested ID
                    parts[2] = str(occupation_id)
                    writer.writerow(parts)
                    total_samples += 1
                else:
                    print(f"Failed after retries: {gender_code}/{occupation_id}/{age}")
                
                # Progress tracking
                if total_samples % 100 == 0:
                    print(f"Generated {total_samples}/3000 samples")
                
                # Rate limiting (adjust as needed)
                time.sleep(0.15)  # ~7 calls/second

print(f"Completed! {total_samples} samples saved to {OUTPUT_FILE}")