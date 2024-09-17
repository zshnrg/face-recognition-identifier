import cv2
import os
import numpy as np
import json
from lib.tempPhoto import TemporaryPhoto
from lib.poseDetector import PoseDetector

class FaceIdentifier:
    def __init__(self, detector='haarcascade_frontalface_default.xml', model_file='face_model.yml', mapping_file='mapping.npy'):
        self.temp_photo = TemporaryPhoto()
        self.pose_detector = PoseDetector()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(detector)
        
        self.model_file = model_file
        self.mapping_file = mapping_file
        self.id_mapping = {}

        # Load model if it exists
        if os.path.exists(self.model_file):
            self.recognizer.read(self.model_file)
            print("Model loaded from", self.model_file)
        
        # Load ID mapping if it exists
        if os.path.exists(self.mapping_file):
            self.id_mapping = np.load(self.mapping_file, allow_pickle=True).item()
            print("ID mapping loaded from", self.mapping_file)

    def __register_images(self, id, name, frames):
        
        if id not in self.id_mapping:
            # Assign a unique integer to this UUID and store the name
            self.id_mapping[id] = {'int_id': len(self.id_mapping) + 1, 'name': name}

        face_id = self.id_mapping[id]['int_id']
        faces = []
        ids = []

        print(f"Processing {len(frames)} frames for user {name} with ID {id}...")

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_detected:
                faces.append(gray[y:y + h, x:x + w])
                ids.append(face_id)  # Use the integer mapping

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(ids, dtype=np.int32))
            self.recognizer.save(self.model_file)  # Save the model

            np.save(self.mapping_file, self.id_mapping)  # Save the ID mapping

            print(f"User {name} registered with ID {id}")
            return True
        else:
            return False

    def __load_registering_user_data(self, id):
        # Load the user data from the temporary folder from the given ID
        try:
            with open(f"temp/{id}/data.json", "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            raise Exception(f"User data not found: {str(e)}")

    def register(self, id, name, frame):
        if self.temp_photo.count(id) < 25:
            # Register the photo
            # Check if directory exists
            if not os.path.exists(f"temp/{id}"):
                os.makedirs(f"temp/{id}")

            try:
                user_data = self.__load_registering_user_data(id)
            except Exception as e:
                user_data = None
            self.pose_detector.set_user_data(user_data)

            res, user_hint = self.pose_detector.capture(frame)

            if not res:
                user_data = self.pose_detector.get_user_data()
                with open(f"temp/{id}/data.json", "w") as f:
                    json.dump(user_data, f)
                
                return {
                    'status': 'Capturing',
                    'hint': user_hint,
                    'progress': self.temp_photo.count(id) / 25
                }
            else:
                self.temp_photo.save(id, frame, f"{user_hint}.jpg")
                user_data = self.pose_detector.get_user_data()

                with open(f"temp/{id}/data.json", "w") as f:
                    json.dump(user_data, f)

                return {
                    'status': 'Capturing',
                    'progress': self.temp_photo.count(id) / 25
                }
        else:
            # Update the model
            print(f"Registering user {name} with ID {id}...")
            frames = self.temp_photo.get(id)
            try:
                res = self.__register_images(id, name, frames)
            except Exception as e:
                return {
                    'status': 'Error',
                    'error': str(e),
                    'progress': 1
                }

            if res:
                # Delete the temporary photos
                os.remove(f"temp/{id}/data.json")
                self.temp_photo.delete(id)
                return {
                    'status': 'Registered',
                    'hint': 'User registered successfully',
                    'progress': 1
                }
            else:
                return {
                    'status': 'Error',
                    'hint': 'Error registering the user',
                    'progress': 1
                }

    def identify(self, id, frame):
        if id not in self.id_mapping:
            raise Exception("User not registered")

        face_id = self.id_mapping[id]['int_id']
        name = self.id_mapping[id]['name']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y + h, x:x + w]
            predicted_id, confidence = self.recognizer.predict(face)

            if predicted_id == face_id and confidence < 50:
                print(f"User identified: \n> ID: {id}\n> Name: {name}\n> Confidence: {round(100 - confidence)}%\n\n")
                return {
                    'status': 'Identified',
                    'name': name,
                    'confidence': round(100 - confidence)
                }
            else:
                print(f"Failed to identify user {id}.\n> Confidence = {round(100 - confidence)}%\n> Face ID: {predicted_id}\n> Expected ID: {face_id}\n\n")
                return {
                    'status': 'Error',
                    'hint': 'User not identified'
                }

        return {
            'status': 'Error',
            'hint': 'No face detected'
        }