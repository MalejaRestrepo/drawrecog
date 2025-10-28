import os
import streamlit as st
import base64
from openai import OpenAI
import openai
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

Expert = " "
profile_imgenh = " "

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except FileNotFoundError:
        return "Error: La imagen no se encontró en la ruta especificada."

# --- Estilo general en tonos fríos ---
st.markdown("""
    <style>
        /* Fondo general con degradado frío */
        .stApp {
            background: linear-gradient(135deg, #e3f2fd 0%, #e8eaf6 50%, #ede7f6 100%);
            font-family: 'Poppins', sans-serif;
            color: #263238;
        }

        /* Título principal */
        h1 {
            background: linear-gradient(90deg, #4b6cb7, #6a85b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.5em;
        }

        /* Subtítulos del panel lateral */
        .stSidebar h2, .stSidebar h3 {
            color: #37474f;
            font-size: 1.2em;
        }

        /* Panel lateral */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e8eaf6 0%, #e3f2fd 100%);
            border-radius: 15px;
            color: #263238;
        }

        /* Texto general oscuro */
        .stMarkdown, .stMarkdown p, label, div[data-testid="stWidgetLabel"], .css-16huue1, .stSlider label {
            color: #263238 !important;
        }

        /* Barra superior degradada */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #6a85b6 0%, #bac8e0 50%, #c5cae9 100%);
            color: white !important;
        }

        /* Sliders personalizados */
        input[type="range"]::-webkit-slider-thumb {
            background: #7e57c2 !important;
            border: 2px solid #5e35b1 !important;
        }

        input[type="range"]::-webkit-slider-runnable-track {
            background: linear-gradient(90deg, #b39ddb, #d1c4e9) !important;
            height: 6px;
            border-radius: 3px;
        }

        /* Canvas con borde azul grisáceo */
        canvas {
            border-radius: 15px !important;
            border: 3px solid #90a4ae;
        }

        /* Texto inferior */
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 1.1em;
            color: #37474f;
        }

        /* Botones */
        button[kind="primary"], button[kind="secondary"], .stButton > button {
            background: linear-gradient(90deg, #7e57c2, #9575cd);
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0px 4px 12px rgba(123, 97, 255, 0.25);
        }

        button[kind="primary"]:hover, button[kind="secondary"]:hover, .stButton > button:hover {
            background: linear-gradient(90deg, #6a5ac7, #7b68ee);
        }

        /* Input de texto */
        .stTextInput input {
            border-radius: 8px !important;
            border: 1px solid #b39ddb !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Configuración principal ---
st.set_page_config(page_title='Tablero Inteligente')
st.title('Tablero Inteligente')

# --- Panel lateral informativo ---
with st.sidebar:
    st.subheader("Acerca de:")
    st.markdown("""
    En esta aplicación se muestra la capacidad de una máquina para interpretar un boceto
    dibujado directamente en el tablero.
    """)

st.subheader("Dibuja el boceto en el panel y presiona el botón para analizarlo")

# --- Configuración del canvas ---
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider('Selecciona el ancho de línea', 1, 30, 5)
stroke_color = "#000000"
bg_color = "#FFFFFF"

canvas_result = st_canvas(
    fill_color="rgba(92, 107, 192, 0.2)",  # Morado frío semitransparente
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=300,
    width=400,
    drawing_mode=drawing_mode,
    key="canvas",
)

# --- Entrada de clave y conexión con OpenAI ---
ke = st.text_input('Ingresa tu Clave')
os.environ['OPENAI_API_KEY'] = ke

api_key = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key)

# --- Botón de análisis ---
analyze_button = st.button("Analiza la imagen", type="secondary")

# --- Lógica de análisis ---
if canvas_result.image_data is not None and api_key and analyze_button:
    with st.spinner("Analizando ..."):
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('img.png')
        
        base64_image = encode_image_to_base64("img.png")
        prompt_text = "Describe en español de forma breve la imagen."

        try:
            full_response = ""
            message_placeholder = st.empty()
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            if response.choices[0].message.content is not None:
                full_response += response.choices[0].message.content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            if Expert == profile_imgenh:
                st.session_state.mi_respuesta = response.choices[0].message.content
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
else:
    if not api_key:
        st.warning("Por favor ingresa tu API key.")

# --- Pie de página ---
st.markdown("""
<div class='footer'>
Explora el poder de la IA para interpretar tus dibujos.
</div>
""", unsafe_allow_html=True)
