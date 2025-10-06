from flask import Flask, render_template, request, send_from_directory, session
from model_compiler import compile_model
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import base64
from io import BytesIO

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='templates', static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super_secret_key'  # Necesario para usar session

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    pred = None
    img_url = None
    # Inicializa historiales si no existen
    if 'image_history' not in session:
        session['image_history'] = []
    if 'draw_history' not in session:
        session['draw_history'] = []
    image_history = session['image_history']
    draw_history = session['draw_history']
    if request.method == 'POST':
        try:
            # Cargar MNIST y compilar modelo fijo solo una vez por sesión
            if 'model_acc' not in session:
                # 1. Carga y Prepara los Datos
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
                x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
                # 2. "Compila" tu Arquitectura
                model = compile_model("Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)", input_shape=(28*28,))
                # 3. Configura y Entrena
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
                # 4. Evalúa el Rendimiento
                loss, acc = model.evaluate(x_test, y_test, verbose=0)
                session['model_acc'] = acc
                model.save_weights('static/model_weights.weights.h5')
            else:
                # Cargar modelo y pesos
                model = compile_model("Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)", input_shape=(28*28,))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                if os.path.exists('static/model_weights.weights.h5'):
                    model.load_weights('static/model_weights.weights.h5')
                acc = session['model_acc']
            result = f"Precisión en MNIST: {acc:.4f}"
            # Procesar imagen subida por archivo
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        os.makedirs(app.config['UPLOAD_FOLDER'])
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(img_path)
                    img = Image.open(img_path).convert('L').resize((28, 28))
                    img_arr = np.array(img)
                    if np.mean(img_arr) > 127:
                        img_arr = 255 - img_arr
                    img_arr = img_arr.astype('float32') / 255.0
                    if np.max(img_arr) < 0.5:
                        img_arr = (img_arr > 0.2).astype('float32')
                    img_arr = img_arr.reshape(1, 28*28)
                    pred_num = int(np.argmax(model.predict(img_arr), axis=1)[0])
                    pred = f"Predicción del número en la imagen: {pred_num}"
                    img_url = f"/static/uploads/{filename}"
                    # Añadir al historial de imágenes
                    image_history.append({'img': img_url, 'num': pred_num})
                    session['image_history'] = image_history
            # Procesar imagen dibujada en canvas
            elif 'draw_image' in request.form and request.form['draw_image']:
                draw_data = request.form['draw_image']
                header, encoded = draw_data.split(',', 1)
                img_bytes = base64.b64decode(encoded)
                img = Image.open(BytesIO(img_bytes)).convert('L').resize((28, 28))
                img_arr = np.array(img)
                if np.mean(img_arr) > 127:
                    img_arr = 255 - img_arr
                img_arr = img_arr.astype('float32') / 255.0
                if np.max(img_arr) < 0.5:
                    img_arr = (img_arr > 0.2).astype('float32')
                img_arr = img_arr.reshape(1, 28*28)
                pred_num = int(np.argmax(model.predict(img_arr), axis=1)[0])
                pred = f"Predicción del número en el dibujo: {pred_num}"
                img_url = draw_data
                # Añadir al historial de dibujos
                draw_history.append({'img': img_url, 'num': pred_num})
                session['draw_history'] = draw_history
        except Exception as e:
            error = f"Error: {str(e)}"
    return render_template(
        'index.html',
        result=result,
        error=error,
        pred=pred,
        img_url=img_url,
        image_history=image_history,
        draw_history=draw_history
    )

if __name__ == '__main__':
    # Asegura que la carpeta templates exista
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Asegura que la carpeta static/uploads exista
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
