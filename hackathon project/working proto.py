from google import genai
from google.genai import types
import json 
import sys # Import sys for a clean exit

# ===== CONFIG =====
# Your API Key is assumed to be correct based on the last attempt.
API_KEY = "" 
IMAGE_PATH_1 = r"C:\Users\John\Desktop\generalbarca\hackathon project\vial51.jpg" # Using vial51/52 as in your last code
IMAGE_PATH_2 = r"C:\Users\John\Desktop\generalbarca\hackathon project\vial52.jpg"

# ===== CREATE CLIENT =====
client = genai.Client(api_key=API_KEY)

# ===== HELPER FUNCTION (Cleaned for production, keeping the file-read check) =====
def load_image(path):
    try:
        with open(path, "rb") as f:
            file_data = f.read()
            # If this prints 0 bytes, the file path is wrong!
            print(f"DEBUG: Successfully read {len(file_data)} bytes from {path}") 
            return types.Part.from_bytes(data=file_data, mime_type="image/jpeg")
    except Exception as e:
        print(f"ERROR: Failed to load image at {path}. Exception: {e}")
        # Stop execution if files can't be read, as the API call will definitely fail
        sys.exit("Critical Error: Image loading failed.") 

# ===== LOAD IMAGES =====
image_part1 = load_image(IMAGE_PATH_1)
image_part2 = load_image(IMAGE_PATH_2)

# ===== RESPONSE SCHEMA =====
schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "directions": types.Schema(
            type=types.Type.STRING,
            description="The exact capitalized medication directions found on the label, starting with an action verb."
        )
    },
    required=["directions"]
)

# ===== PROMPT (Updated for JSON output) =====
prompt = """
From the two medication vial and label pictures provided, find and extract the name of the medication the dose, the instruction, and duration.
The instructions are fully capitalized and start with an action verb (e.g., TAKE, APPLY). 
Output ONLY the resulting JSON object that matches the provided schema.
"""

# ===== CALL GEMINI =====
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        # Send prompt and images together in the contents list
        contents=[prompt, image_part1, image_part2], 
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,                
            max_output_tokens=150,
            temperature=0.0
        )
    )

    # *** CRITICAL DEBUG STEP: Print the raw text returned by the API ***
    print(f"\nDEBUG: Raw API Response Text: '{response.text}'") 
    # *********************************************************

    # Check for empty response text
    if not response.text or response.text.strip() == "":
        print("\nFATAL API ERROR: The API returned an empty response. This is usually due to a **Quota Limit** being hit or a **revoked API Key**.")
        sys.exit()

    # ===== GET THE RESULT =====
    # If the text is non-empty, we try to load the JSON
    data = json.loads(response.text)
    directions_text = data.get("directions", "No directions key found in valid JSON.")

    print("\nMedication directions:")
    print(directions_text.strip())

except Exception as e:
    print(f"\nError during API call or JSON parsing: {e}")