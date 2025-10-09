import os
import base64
import requests
import json
import glob
import csv
from PIL import Image
import io

def get_api_key():
    """
    Retrieves the Gemini API key from an environment variable.
    Raises an error if the key is not found.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your Google AI API key.")
    return api_key

def image_to_base64(image_path):
    """
    Converts an image file to a base64 encoded string and gets its MIME type.
    """
    try:
        with Image.open(image_path) as img:
            # Determine MIME type from image format
            format = img.format
            if format == 'JPEG':
                mime_type = 'image/jpeg'
            elif format == 'PNG':
                mime_type = 'image/png'
            else:
                # Fallback for other types, save to a buffer to get bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                mime_type = 'image/png'
                img_bytes = buffer.getvalue()
            
            if 'img_bytes' not in locals():
                 with open(image_path, "rb") as image_file:
                    img_bytes = image_file.read()

        base64_string = base64.b64encode(img_bytes).decode('utf-8')
        return base64_string, mime_type
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def process_image_with_ai(base64_image, mime_type, api_key):
    """
    Sends the image data to the Google Gemini AI for processing.
    Returns the extracted data as a list of dictionaries.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    prompt = """
    From the provided image of a leaderboard, extract the names and their corresponding scores. 
    Return the result as a JSON array of objects, where each object has a 'name' and a 'number' key. 
    For example: [{ "name": "Player1", "number": 123 }]. 
    Do not include any other text, explanations, or markdown formatting in your response. Just the raw JSON array.
    """

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime_type, "data": base64_image}}
            ]
        }],
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        result = response.json()
        candidate = result.get('candidates', [{}])[0]
        content_part = candidate.get('content', {}).get('parts', [{}])[0]
        
        if 'text' in content_part:
            response_text = content_part['text']
            # Clean the response to ensure it's valid JSON
            cleaned_json_string = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_json_string)
        else:
            print("Warning: AI response did not contain text data.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from AI response: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return []


def write_to_csv(data, filename="leaderboard_data_combined.csv"):
    """
    Writes the extracted data to a CSV file with UTF-8 encoding (with BOM).
    """
    if not data:
        print("No data to write to CSV.")
        return
        
    # Using 'utf-8-sig' encoding to add a BOM, which helps Excel open the file correctly
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Name', 'Number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')

        writer.writeheader()
        # Rename keys to match fieldnames for DictWriter
        for item in data:
            writer.writerow({'Name': item.get('name'), 'Number': item.get('number')})

    print(f"\nSuccessfully wrote {len(data)} records to {filename}")


def main():
    """
    Main function to orchestrate the image processing and CSV export.
    """
    print("--- AI Leaderboard Exporter ---")
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        return

    folder_path = input("Enter the path to the folder containing your images: ")
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    # Find all common image types
    image_paths = glob.glob(os.path.join(folder_path, '*.png')) + \
                  glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg'))

    if not image_paths:
        print(f"No images found in '{folder_path}'.")
        return

    print(f"Found {len(image_paths)} images to process.")
    all_extracted_data = []

    for i, path in enumerate(image_paths):
        print(f"\nProcessing image {i + 1}/{len(image_paths)}: {os.path.basename(path)}...")
        base64_image, mime_type = image_to_base64(path)
        
        if base64_image and mime_type:
            extracted_data = process_image_with_ai(base64_image, mime_type, api_key)
            if extracted_data:
                all_extracted_data.extend(extracted_data)
                print(f"  -> Extracted {len(extracted_data)} records.")
            else:
                print("  -> No records extracted from this image.")

    write_to_csv(all_extracted_data)

if __name__ == "__main__":
    main()
