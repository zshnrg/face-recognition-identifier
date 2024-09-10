import os
import face_recognition
import cv2

class FaceIdentifier:

    __imagesPath = './images'

    def __init__(self):
        self.fr = face_recognition

        if not os.path.exists(self.__imagesPath):
            os.makedirs(self.__imagesPath)

    def __loadModel(self, id):

        if not os.path.exists(f'{self.__imagesPath}/{id}'):
            return [], []

        known_face_encodings = []
        known_face_names = []

        for file in os.listdir(f'{self.__imagesPath}/{id}'):
            image = self.fr.load_image_file(f'{self.__imagesPath}/{id}/{file}')
            # File format: id_name_[number].jpg
            name = file.split('_')[1]
            
            known_face_encodings.append(self.fr.face_encodings(image)[0])
            known_face_names.append(name)

        return known_face_encodings, known_face_names

    def register(self, id, name, images):
        if not os.path.exists(f'{self.__imagesPath}/{id}'):
            os.makedirs(f'{self.__imagesPath}/{id}')

        for i, img in enumerate(images):
            cv2.imwrite(f'{self.__imagesPath}/{id}/{id}_{name}_{i}.jpg', img)

    def identify(self, id, img):
        known_face_encodings, known_face_names = self.__loadModel(id)
        if not known_face_encodings:
            return 'No images registered'

        face_locations = self.fr.face_locations(img)
        face_encodings = self.fr.face_encodings(img, face_locations)

        for face_encoding in face_encodings:
            matches = self.fr.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            return name
        
        

    


        