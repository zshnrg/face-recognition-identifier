from lib.faceIdentifier import FaceIdentifier
from flask import Flask, request, render_template, Response, jsonify
import uuid 
import cv2
import os

app = Flask(__name__)

face_identifier = FaceIdentifier()

global camera
camera = cv2.VideoCapture(0)

# API routes

@app.route('/api/register', methods=['POST'])
def __register():
    app.logger.info('Registering images')
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    name = request.form.get('name')
    id = str(uuid.uuid4())

    try:
        face_identifier.register(id, name, files)
    except Exception as e:
        app.logger.error('Error registering the images')
        return jsonify({'Error registering the images': str(e)}), 500
    
    app.logger.info('Images registered successfully')
    return jsonify({'Success': 'Images registered successfully', 'id': id}), 200

@app.route('/api/identify', methods=['POST'])
def __identify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img = request.files['image']
    id = request.form.get('id')

    try:
        name = face_identifier.identify(id, img)
    except Exception as e:
        return jsonify({'Error identifying the image': str(e)}), 500

    return jsonify({'name': name}), 200

# Web routes

def gen_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    app.logger.info('Accessing home page')
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    app.logger.info('Accessing video feed')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verify', methods=['POST'])
def verify():
    # Check if the user id not empty and if the user id is valid
    id = request.form.get('id')
    if not id:
        app.logger.error('No id provided')
        return render_template('home.html', error_messages=['No ID provided'])
    
    if not os.path.exists(f'./images/{id}'):
        app.logger.error('Invalid id')
        return render_template('home.html', error_messages=['Invalid ID, please register first'])

    app.logger.info('Accessing verify page')
    return render_template('verify.html')

@app.route('/register', methods=['POST'])
def register():
    app.logger.info('Accessing register page')
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)

cv2.destroyAllWindows()