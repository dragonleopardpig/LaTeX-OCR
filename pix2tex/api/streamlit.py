import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from st_img_pastebutton import paste


def encode_image(file):
    _, encoded = file.split(",", 1)
    binary_data = base64.b64decode(encoded)
    bytes_data = BytesIO(binary_data)
    return bytes_data


if __name__ == "__main__":
    st.set_page_config(page_title="LaTeX-OCR")
    st.title("LaTeX OCR")
    st.markdown(
        "Convert images of equations to corresponding LaTeX code.\n\nThis is based on the `pix2tex` module. Check it out [![github](https://img.shields.io/badge/LaTeX--OCR-visit-a?style=social&logo=github)](https://github.com/lukas-blecher/LaTeX-OCR)"
    )

    source = st.radio(
        "Choose the source of the image",
        options=["Upload", "Paste"],
    )

    image = None

    if source == "Upload":
        uploaded_file = st.file_uploader(
            "Upload an image of an equation",
            type=["png", "jpg"],
        )

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file))
            image = uploaded_file.getvalue()

    if source == "Paste":
        pasted_file = paste("Paste an image of an equation")

        if pasted_file is not None:
            image = encode_image(pasted_file)
            st.image(image)

    if st.button("Convert"):
        if image is not None:
            with st.spinner("Computing"):
                try:
                    api_key = os.environ.get('PIX2TEX_API_KEY')
                    response = requests.post(
                        "http://127.0.0.1:8502/predict/",
                        files={"file": image},
                        headers={'X-API-Key': api_key} if api_key else None,
                        timeout=180,
                    )
                except requests.RequestException as error:
                    st.error(f"The local OCR service is unavailable: {error}")
                    st.stop()
            if response.ok:
                latex_code = response.json()
                st.code(latex_code, language="latex")
                st.latex(latex_code)
            else:
                st.error(response.text)
        else:
            st.error("No image selected")
