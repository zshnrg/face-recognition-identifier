from lib.faceIdentifier import FaceIdentifier
from flask import Flask, request, render_template, Response, jsonify
import uuid 
import cv2
import os
import numpy as np

app = Flask(__name__)

face_identifier = FaceIdentifier()

# API routes

@app.route('/api/v1/register', methods=['POST'])
def __register():
    app.logger.info('Registering images')
    if 'image' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    img = request.files['image']
    name = request.form.get('name')
    id = request.form.get('id')

    # Convert the image to numpy arrays
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        res = face_identifier.register(id, name, img)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    return jsonify(res), 200

@app.route('/api/v1/identify', methods=['POST'])
def __identify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img = request.files['image']
    id = request.form.get('id')

    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        res = face_identifier.identify(id, img)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(res), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run()