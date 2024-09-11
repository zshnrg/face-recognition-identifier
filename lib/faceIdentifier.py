import cv2
import os
import numpy as np

class FaceIdentifier:
    def __init__(self, detector='haarcascade_frontalface_default.xml', model_file='face_model.yml', mapping_file='mapping.npy'):
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

    def register(self, id, name, frames):
        if id not in self.id_mapping:
            # Assign a unique integer to this UUID and store the name
            self.id_mapping[id] = {'int_id': len(self.id_mapping) + 1, 'name': name}

        face_id = self.id_mapping[id]['int_id']
        faces = []
        ids = []

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
        else:
            print("No face detected in the provided frames.")

    def identify(self, id, frame):
        if id not in self.id_mapping:
            print("User ID not found")
            return False

        face_id = self.id_mapping[id]['int_id']
        name = self.id_mapping[id]['name']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

        for (x, y, w, h) in faces_detected:
            face = gray[y:y + h, x:x + w]
            predicted_id, confidence = self.recognizer.predict(face)

            if predicted_id == face_id and confidence < 50:
                print(f"User identified: \n> ID: {id}\n> Name: {name}\n> Confidence: {round(100 - confidence)}%\n\n")
                return name
            else:
                print(f"Failed to identify user {id}.\n> Confidence = {round(100 - confidence)}%\n> Face ID: {predicted_id}\n> Expected ID: {face_id}\n\n")
                return None

        print("No face detected in the frame.")
        return None