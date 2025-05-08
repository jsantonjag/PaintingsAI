# Instalación de la libreríia duckduckgo-search
pip install duckduckgo-search pillow
pip install duckduckgo-search --upgrade

# Carga de librerías
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS

# Cargar el dataset original
url = "https://raw.githubusercontent.com/jsantonjag/PaintingsAI/refs/heads/main/data/dataset_completo.csv"
dataset = pd.read_csv(url)

# Carpeta donde se guardarán las imágenes
output_dir = "dowloaded_images"
os.makedirs(output_dir, exist_ok=True)

with DDGS() as ddgs:
    for index, row in dataset.iterrows():
        query = f"{row['picture'].replace('-', ' ').replace('.jpg', '')} {row['artist']} {row['style']}"
        print(f"[{index+1}] Buscando: {query}")

        try:
            results = ddgs.images(keywords=query, max_results=1)
            result_list = list(results)
            if result_list:
                url = result_list[0]["image"]
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                
                # Guardar la imagen en la carpeta 
                filename = os.path.join(output_dir, row['picture'])
                img.save(filename)

                print(f" Guardada: {filename}")
            else:
                print(f" No se encontraron resultados para: {query}")
        except Exception as e:
            print(f" Error al procesar {query}: {e}")



