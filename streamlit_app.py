import streamlit as st
import requests
import base64
import json
import pandas as pd
from io import BytesIO

# --- Configuration and Initialization ---

# Set up the page title and layout
st.set_page_config(layout="wide", page_title="AI Multi-Image Exporter")
st.title("AI Multi-Image Exporter")
st.markdown("Upload multiple images, extract data with Google AI, and export to a single CSV.")

# Initialize session state for data storage
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
if 'files_to_process' not in st.session_state:
    st.session_state.files_to_process = []
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Ready to process images."

# --- API Key Setup (Mimicking JS logic for local use) ---
# In a real deployed Streamlit app, you would use st.secrets['GEMINI_API_KEY']
API_KEY = "AIzaSyCzLdSYJFl7WsfZjPvOmhz11FDJJZxunWU"  # REPLACE THIS EMPTY STRING WITH YOUR GEMINI API KEY FOR LOCAL TESTING

# --- Functions ---

def convert_to_base64_and_mime(uploaded_file):
    """Converts a Streamlit uploaded file object to Base64 and extracts MIME type."""
    try:
        # Go to the start of the file before reading
        uploaded_file.seek(0)
        
        # Read the file content as bytes
        image_bytes = uploaded_file.read()
        
        # Encode bytes to Base64 string
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        
        # Get MIME type directly from the file object
        mime_type = uploaded_file.type
        
        return mime_type, base64_data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

def call_gemini_api(mime_type, base64_data, api_key):
    """Makes the API call to the Gemini model."""
    if not api_key:
        raise ValueError("API Key is missing. Please provide your Gemini API key.")
        
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
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
                {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            ]
        }],
    }

    # API call with a simple retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Raises an HTTPError if the status is 4xx or 5xx
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                # Specific check for 403 Forbidden (API Key issue)
                raise PermissionError("API call failed with status 403 Forbidden. Check your API key.")
            
            # General error handling and exponential backoff
            last_error = e
            delay = 2 ** attempt
            if attempt < max_retries - 1:
                import time
                time.sleep(delay)
        except requests.exceptions.RequestException as e:
            last_error = e
            delay = 2 ** attempt
            if attempt < max_retries - 1:
                import time
                time.sleep(delay)
    
    raise last_error # Raise the last encountered error

def parse_and_store_results(result):
    """Extracts and cleans the JSON result from the API response."""
    candidate = result.get('candidates', [{}])[0]
    response_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
    
    # Clean up potential markdown wrappers (```json ... ```)
    cleaned_json_string = response_text.strip().replace('```json', '').replace('```', '').strip()
    
    try:
        data = json.loads(cleaned_json_string)
        if isinstance(data, list):
            st.session_state.extracted_data.extend(data)
        else:
            st.warning(f"AI response was not a JSON list: {cleaned_json_string}")
    except json.JSONDecodeError:
        st.error(f"Failed to parse JSON from AI response: {cleaned_json_string[:100]}...")

def process_images():
    """Main processing logic for all uploaded images."""
    st.session_state.extracted_data = []
    error_count = 0
    
    if not st.session_state.files_to_process:
        st.session_state.status_message = 'Please upload one or more images first.'
        return

    # Use Streamlit's status container for visual feedback
    with st.status("Processing images...", expanded=True) as status_box:
        
        for i, uploaded_file in enumerate(st.session_state.files_to_process):
            st.write(f"Processing image {i + 1} of {len(st.session_state.files_to_process)}: {uploaded_file.name}")
            
            # 1. Convert to Base64
            mime_type, base64_data = convert_to_base64_and_mime(uploaded_file)
            if not base64_data:
                error_count += 1
                continue
            
            # 2. Call the API
            try:
                result = call_gemini_api(mime_type, base64_data, API_KEY)
                # 3. Parse and Store Results
                parse_and_store_results(result)
            except PermissionError as e:
                st.error(str(e))
                status_box.update(label="Processing Failed", state="error", expanded=True)
                return
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                error_count += 1

        # 4. Final Status Update
        total_records = len(st.session_state.extracted_data)
        success_count = len(st.session_state.files_to_process) - error_count
        
        final_status = f"Found a total of {total_records} records from {success_count} successfully processed image(s)."
        if error_count > 0:
            final_status += f" ({error_count} image(s) failed to process.)"
            status_box.update(label=f"Processing Complete with Errors: {final_status}", state="error", expanded=False)
        else:
            status_box.update(label=f"Processing Complete: {final_status}", state="complete", expanded=False)
            
        st.session_state.status_message = final_status

# --- UI Layout ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Image Upload & Controls")
    
    # 1. Image Uploader (Replacing the hidden input and file reader)
    uploaded_files = st.file_uploader(
        "Upload Image(s)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True,
        help="Upload images containing names and scores (e.g., leaderboards)."
    )

    if uploaded_files:
        if uploaded_files != st.session_state.files_to_process:
            st.session_state.files_to_process = uploaded_files
            st.session_state.status_message = f"{len(uploaded_files)} image(s) loaded. Ready to process."
        
        # Display image previews (similar to the JS version)
        st.caption("Image Previews:")
        preview_cols = st.columns(min(len(uploaded_files), 5)) # Show up to 5 previews per row
        for i, file in enumerate(uploaded_files):
            # Show a smaller preview
            preview_cols[i % 5].image(file, caption=file.name, use_column_width=True)

    # 2. Process Button
    st.button(
        "Process Images with AI",
        on_click=process_images,
        disabled=not st.session_state.files_to_process,
        use_container_width=True,
        type="primary"
    )
    
    # 3. Status Display
    st.info(st.session_state.status_message)


with col2:
    st.subheader("Combined Results")
    
    if st.session_state.extracted_data:
        # 1. Display Results in a DataFrame
        df = pd.DataFrame(st.session_state.extracted_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # 2. Export Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name="leaderboard_data_combined.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown(
            """
            <div style="padding: 100px; text-align: center; color: #9ca3af;">
                Process image(s) to see results.
            </div>
            """, 
            unsafe_allow_html=True
        )

# --- API Key Warning for Local Testing (mimics 403 error logic) ---
if not API_KEY:
    st.warning("⚠️ **API Key Warning**")
    st.markdown(
        """
        Since you are running this locally, you must replace the empty string for `API_KEY` 
        in `streamlit_app.py` with your actual Gemini API Key to avoid a 403 Forbidden error.
        
        For deployment, use Streamlit Secrets instead of hardcoding the key.
        """
    )
