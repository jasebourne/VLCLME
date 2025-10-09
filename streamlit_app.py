import streamlit as st
import base64
import requests
import json
import pandas as pd

def image_to_base64(uploaded_file):
    """
    Converts an uploaded file object (from Streamlit) to a base64 encoded string.
    """
    img_bytes = uploaded_file.getvalue()
    base64_string = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = uploaded_file.type
    return base64_string, mime_type

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
        response.raise_for_status()
        
        result = response.json()
        candidate = result.get('candidates', [{}])[0]
        content_part = candidate.get('content', {}).get('parts', [{}])[0]
        
        if 'text' in content_part:
            response_text = content_part['text']
            cleaned_json_string = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_json_string)
        else:
            st.warning("AI response did not contain text data for one of the images.")
            return []

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except json.JSONDecodeError:
        st.error(f"Failed to decode JSON from AI response: {response.text}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    
    return []

def main():
    """
    Main function to define the Streamlit application UI and logic.
    """
    st.set_page_config(layout="wide")
    st.title("🖼️ AI Leaderboard Exporter")
    st.markdown("Upload multiple leaderboard images, extract data with Google AI, and download the results as a single CSV file.")

    with st.sidebar:
        st.header("🔑 API Key")
        api_key = st.text_input("Enter your Google AI API Key", type="password")
        st.markdown("[Get an API key from Google AI Studio](https://aistudio.google.com/app/apikey)")

    uploaded_files = st.file_uploader(
        "Upload your leaderboard images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.image(uploaded_files, width=150, caption=[f.name for f in uploaded_files])

    if st.button("🚀 Process Images"):
        if not api_key:
            st.warning("Please enter your Google AI API key in the sidebar.")
        elif not uploaded_files:
            st.warning("Please upload one or more images.")
        else:
            all_extracted_data = []
            my_bar = st.progress(0, text="Starting processing...")

            for i, uploaded_file in enumerate(uploaded_files):
                my_bar.progress((i + 1) / len(uploaded_files), text=f"Processing: {uploaded_file.name}")
                
                base64_image, mime_type = image_to_base64(uploaded_file)
                extracted_data = process_image_with_ai(base64_image, mime_type, api_key)
                if extracted_data:
                    all_extracted_data.extend(extracted_data)

            my_bar.empty()

            if all_extracted_data:
                st.success(f"Successfully extracted {len(all_extracted_data)} records from {len(uploaded_files)} images!")
                
                # Use pandas for robust data handling and CSV conversion
                df = pd.DataFrame(all_extracted_data)
                df.rename(columns={'name': 'Name', 'number': 'Number'}, inplace=True)
                st.dataframe(df)

                # Convert dataframe to CSV string, using utf-8-sig for Excel compatibility
                csv = df.to_csv(index=False).encode('utf-8-sig')

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="leaderboard_data_combined.csv",
                    mime="text/csv",
                )
            else:
                st.error("Could not extract any data from the uploaded images. Please try with clearer images.")

if __name__ == "__main__":
    main()

