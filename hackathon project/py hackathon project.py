import os
import glob
import sys
import json
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---

# IMPORTANT: Best practice is to set your API key as an environment variable (GEMINI_API_KEY).
# If you must hardcode it for testing, replace the placeholder below.
# API_KEY = "AIzaSy..." 
API_KEY = os.getenv("GEMINI_API_KEY", "")

# Define your image paths here. The script will accept 1, 2, or 3 paths.
# You MUST update these paths to point to your local images (e.g., vial51.jpg, vial52.jpg).

# 1. Define the base directory where your images are located
# NOTE: Using a raw string (r"...") is good practice for Windows paths.
IMAGE_DIRECTORY = r"C:\Users\John\Desktop\generalbarca\hackathon project"

# 2. Define the list of common image extensions you want to search for
# We will search for both lowercase and uppercase extensions to be safe.
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', 
                    '*.JPG', '*.JPEG', '*.PNG', '*.GIF', '*.BMP', '*.TIFF']

# 3. Initialize an empty list to store the final, collected image paths
IMAGE_PATHS = []

# 4. Loop through each desired extension and use glob to find matching files
for ext in IMAGE_EXTENSIONS:
    # os.path.join safely constructs the full path pattern: 
    # C:\Users\John\Desktop\...\hackathon project\*.jpg
    search_pattern = os.path.join(IMAGE_DIRECTORY, ext)
    
    # glob.glob finds all files matching the pattern and adds them to the list
    IMAGE_PATHS.extend(glob.glob(search_pattern))

# --- Verification (Optional, but highly recommended) ---
print(f"Searching in directory: {IMAGE_DIRECTORY}")
if IMAGE_PATHS:
    print(f"Successfully found {len(IMAGE_PATHS)} image file(s).")
    for path in IMAGE_PATHS:
        print(f" - {path}")
else:
    print("WARNING: No image files were found in the specified directory.")

# --- 2. HELPER FUNCTION ---

def load_images_as_parts(paths):
    """
    Loads one or more image files from local paths and converts them into 
    Gemini API Part objects for the contents list.
    """
    if not paths:
        sys.exit("ERROR: No image paths provided in IMAGE_PATHS list.")
    
    image_parts = []
    print(f"Loading {len(paths)} image(s)...")
    
    for i, path in enumerate(paths):
        try:
            with open(path, "rb") as f:
                file_data = f.read()
            
            # Use 'image/jpeg' or 'image/png' based on your file type
            part = types.Part.from_bytes(data=file_data, mime_type="image/jpeg")
            image_parts.append(part)
            print(f"DEBUG: Successfully read {len(file_data)} bytes from image {i+1}: '{os.path.basename(path)}'")
            
        except FileNotFoundError:
            print(f"FATAL ERROR: File not found at path: {path}")
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: Failed to load image at {path}. Exception: {e}")
            sys.exit(1)
            
    return image_parts


# --- 3. RESPONSE SCHEMA DEFINITION ---

# Define the exact structure of the JSON output we require from the model.
schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "medication_name": types.Schema(
            type=types.Type.STRING,
            description="The full, unambiguous name of the medication (e.g., Lipitor, Amoxicillin)."
        ),
        "dosage_strength": types.Schema(
            type=types.Type.STRING,
            description="The strength and form of the medication (e.g., 500 MG CAP, 2.5mg Tablet)."
        ),
        "directions": types.Schema(
            type=types.Type.STRING,
            description="The exact, fully capitalized instructions for use from the label, starting with an action verb (e.g., TAKE 1 CAPSULE TWICE DAILY)."
        )
    },
    # Ensure all these keys must be present in the final JSON output
    required=["medication_name", "dosage_strength", "directions"] 
)

# --- 4. MAIN LOGIC ---

def run_extractor():
    """Initializes the client, loads data, calls the API, and processes the result."""
    
    # Initialize Client
    if not API_KEY:
        sys.exit("ERROR: API_KEY is missing. Please set the GEMINI_API_KEY environment variable or hardcode it.")
        
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        sys.exit(f"ERROR: Could not initialize Gemini client: {e}")

    # Load all images
    image_parts = load_images_as_parts(IMAGE_PATHS)

    # Construct the final prompt
    # The prompt is constructed to be strict about the structured output.
    prompt = """
    Analyze the provided images of medication labels. Your task is to extract three specific pieces of information.
    1. Medication Name
    2. Medication Dose/Strength
    3. Directions for Use (should be fully capitalized and start with an action verb).
    
    Strictly ensure all property names and string values are enclosed in double quotes.
    Output ONLY the resulting JSON object that strictly matches the provided schema.
    """

    # Prepare contents list: Prompt first, then all image parts
    contents = [prompt] + image_parts

    # Call Gemini API
    print("\nCalling Gemini API...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents, 
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema, 
                max_output_tokens=2048, # Increased for safety
                temperature=0.0
            )
        )

        # DEBUG: Print the raw text returned by the API
        print(f"DEBUG: Raw API Response Text: '{response.text}'") 

        if not response.text or response.text.strip() == "":
            print("\nFATAL API ERROR: The API returned an empty response (likely Quota Limit or a bad key).")
            return

        # Parse JSON
        data = json.loads(response.text)
        
        # Display Results
        print("\n--- ✅ Extracted Medication Data ---")
        print(f"Medication Name    : {data.get('medication_name', 'N/A')}")
        print(f"Dosage Strength    : {data.get('dosage_strength', 'N/A')}")
        print(f"Directions         : {data.get('directions', 'N/A')}")
        print("-----------------------------------")

    except json.JSONDecodeError as e:
        print(f"\nFATAL PARSING ERROR: The API returned malformed JSON (truncated or incorrect format).")
        print(f"Error: {e}")
        print("Check 'Raw API Response Text' above for clues.")
    except Exception as e:
        print(f"\nFATAL API ERROR: An unexpected error occurred during the API call: {e}")

# Execute the main function
if __name__ == "__main__":
    run_extractor()