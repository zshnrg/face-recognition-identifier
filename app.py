from lib.faceIdentifier import FaceIdentifier
from flask import Flask, request, jsonify
import uuid 

app = Flask(__name__)

face_identifier = FaceIdentifier()

# API routes

@app.route('/api/register', methods=['POST'])
def register():
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    name = request.form.get('name')
    id = str(uuid.uuid4())

    try:
        face_identifier.register(id, name, files)
    except Exception as e:
        return jsonify({'Error registering the images': str(e)}), 500
    
    return jsonify({'Success': 'Images registered successfully', 'id': id}), 200

@app.route('/api/identify', methods=['POST'])
def identify():
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

@app.route('/')
def index():
    return '''
        <h1>Face Identifier</h1>
        <button onclick="register()">Register</button>
    '''

if __name__ == '__main__':
    app.run()