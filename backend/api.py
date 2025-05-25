from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
import joblib
import pandas as pd

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar la clave de API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


modelo = joblib.load('modelo.pkl')
columnas_modelo = joblib.load('modelo_columnas.pkl')

app = Flask(__name__)

@app.route('/chatgpt', methods=['POST'])
def chatgpt():
    try:
        # Obtener datos del cliente
        data = request.get_json()
        tdp = data.get('tdp', 0)
        process_size = data.get('processSize', 0)
        freq = data.get('freq', 0)
        producto = data.get('producto', '')
        tipo = data.get('tipo', '')
        vendedor = data.get('vendedor', '')
        carrera = data.get('carrera', '')  # Nuevo dato recibido

        # Generar el prompt
        prompt = (
            f"Un estudiante de la carrera {carrera} está buscando un componente de hardware. "
            f"El componente tiene las siguientes características: TDP {tdp}W, "
            f"proceso {process_size}nm, frecuencia {freq}MHz, "
            f"producto {producto}, tipo {tipo}, y vendedor {vendedor}. "
            f"¿Cuál es el precio estimado de este componente en dólares y pesos mexicanos? "
            f"Además, ¿es este componente adecuado para las necesidades de la carrera {carrera}? "
            f"Dame una respuesta corta y concisa, e indica si el producto es de gama baja, media o alta."
        )
        print(f"Prompt generado: {prompt}")  # Depuración

        # Llamar a la API de OpenAI con el método correcto
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Cambia el modelo si es necesario
            messages=[
                {"role": "system", "content": "Eres un asistente experto en hardware."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150  # Ajusta según la longitud esperada de la respuesta
        )

        # Extraer la respuesta correctamente
        answer = response.choices[0].message.content.strip()

        return jsonify({"price": answer})

    except Exception as e:
        print("Error en /chatgpt:", str(e))  # Depuración
        return jsonify({"error": "Error interno del servidor"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del cliente
        data = request.get_json()
        entrada = pd.DataFrame([{
            'Product': data['producto'],
            'Type': data['tipo'],
            'Vendor': data['vendedor'],
            'TDP (W)': data['tdp'],
            'Process Size (nm)': data['processSize'],
            'Freq (MHz)': data['freq']
        }])

        # Convertir columnas categóricas a variables dummy
        entrada = pd.get_dummies(entrada, columns=['Product', 'Type', 'Vendor'], drop_first=True)

        # Identificar columnas faltantes
        columnas_faltantes = [col for col in columnas_modelo if col not in entrada]

        # Crear un DataFrame con las columnas faltantes y valores 0
        faltantes_df = pd.DataFrame(0, index=entrada.index, columns=columnas_faltantes)

        # Concatenar las columnas faltantes al DataFrame original
        entrada = pd.concat([entrada, faltantes_df], axis=1)

        # Ordenar columnas según el modelo
        entrada = entrada[columnas_modelo]

        # Realizar predicción
        resultado = modelo.predict(entrada)[0]
        if (resultado < 2000):
            gama = "baja"
        elif (resultado >= 2000 and resultado < 5000):
            gama = "media"
        else:
            gama = "alta"


        return jsonify({'resultado': resultado, 'gama': gama})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)