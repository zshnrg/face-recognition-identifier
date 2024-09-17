import os
import cv2

class TemporaryPhoto:
    def __init__(self, source="temp/"):
        # Class constructor that sets the source directory where the photos will be stored
        # Create the source directory if it doesn't exist
        self.source = source
        try:
            if not os.path.exists(source):
                os.makedirs
        except Exception as e:
            raise Exception(f"Error creating the source directory: {str(e)}")

    def save(self, id, frame, filename):
        # Save the photo with the given ID to a folder
        try:
            path = os.path.join(self.source, id)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, filename), frame)

            return True
        except Exception as e:
            raise Exception(f"Error saving the photo: {str(e)}")

    def delete(self, id):
        # Delete all the photos with the given ID from the folder
        try:
            path = os.path.join(self.source, id)
            for photo in os.listdir(path):
                os.remove(os.path.join(path, photo))
            os.rmdir(path)

            return True
        except Exception as e:
            raise Exception(f"Error deleting the photo: {str(e)}")

    def count(self, id):
        # Count the number of photos with the given ID in the folder (check file extension, there is json file)
        # Return 0 if the folder doesn't exist
        if not os.path.exists(os.path.join(self.source, id)):
            return 0
        try:
            path = os.path.join(self.source, id)
            return len([photo for photo in os.listdir(path) if photo.endswith(".jpg")])
        except Exception as e:
            raise Exception(f"Error counting the photos: {str(e)}")

    def get(self, id):
        # Return array of photos with the given ID from the folder
        try:
            path = os.path.join(self.source, id)
            images = []
            for photo in os.listdir(path):
                if photo.endswith(".jpg"):
                    images.append(cv2.imread(os.path.join(path, photo)))
            return images
        except Exception as e:
            raise Exception(f"Error getting the photos: {str(e)}")

    