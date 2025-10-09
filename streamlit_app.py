
import numpy as np
import cv2
import pytesseract
import re
import pandas as pd
import streamlit as st
import io

st.title('Name and Number Extractor from Image')
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

def extract_name_number_pairs(text):
    # Refined regex to handle variations and Unicode letters
    # \w with re.UNICODE will match accented characters
    pattern = re.compile(r'([\w\s.-]+)\s*(\d+)', re.UNICODE)
    pairs = pattern.findall(text)
    cleaned_pairs = [(name.strip(), number) for name, number in pairs]
    return cleaned_pairs


if uploaded_file is not None:
    try:
        # Read the uploaded file as bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Decode the image using OpenCV
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Basic image preprocessing (Convert to grayscale and apply a simple threshold)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extract text using pytesseract on the preprocessed image
        # Use lang='eng' by default, but you can add more languages if needed (e.g., 'eng+fra+deu')
        extracted_text = pytesseract.image_to_string(thresh_img, lang='eng')
        st.write("Extracted Text:")
        st.write(extracted_text)

        # Parse extracted text
        name_number_pairs = extract_name_number_pairs(extracted_text)

        # Check if any pairs were found
        if name_number_pairs:
            st.write("Identified Name-Number Pairs:")
            df_results = pd.DataFrame(name_number_pairs, columns=['Name', 'Number'])
            st.dataframe(df_results)
        else:
            st.write("No name-number pairs found in the extracted text.")
            df_results = pd.DataFrame(columns=['Name', 'Number']) # Create empty DF even if no pairs

    except Exception as e:
        st.error(f"An error occurred during image processing or text extraction: {e}")
        df_results = pd.DataFrame(columns=['Name', 'Number']) # Ensure df_results is defined on error
else:
    # Ensure df_results is defined when no file is uploaded
    df_results = pd.DataFrame(columns=['Name', 'Number'])

# Display the DataFrame if it's not empty (will be empty on error or no pairs found)
if 'df_results' in locals() and not df_results.empty:
    st.subheader("Extracted Name and Number Data:")
    st.dataframe(df_results)
    # Add download button for UTF-8 CSV export
    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False, encoding='utf-8')
    st.download_button(
        label="Download results as CSV",
        data=csv_buffer.getvalue(),
        file_name="name_number_results.csv",
        mime="text/csv",
    )
elif 'df_results' in locals() and df_results.empty:
    # This case is handled by the "No name-number pairs found" message or the error message,
    # so no extra message is needed here.
    pass