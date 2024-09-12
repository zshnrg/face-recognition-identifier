from lib.faceIdentifier import FaceIdentifier
from flask import Flask, request, render_template, Response, jsonify
import uuid 
import cv2
import os
import numpy as np

app = Flask(__name__)

face_identifier = FaceIdentifier()

# API routes

@app.route('/api/register', methods=['POST'])
def __register():
    app.logger.info('Registering images')
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    name = request.form.get('name')
    id = str(uuid.uuid4())

    # Convert the images to numpy arrays
    files = [cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) for file in files]

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

    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        name = face_identifier.identify(id, img)
    except Exception as e:
        return jsonify({'Error identifying the image': str(e)}), 500

    return jsonify({'name': name}), 200

if __name__ == '__main__':
    app.run(debug=True)