import mediapipe as mp
import numpy as np
import time
import cv2

class PoseDetector:
    def __init__(self):
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        
        self.user_data = None
        self.last_capture_time = time.time()
        self.captured_images = []

        self.offset_x = 0
        self.offset_y = 0

        self.directions = [
            [(0, 20), (10, 20), (20, 20), (20, 10), (20, 0)],
            [(20, -10), (20, -20), (10, -20), (0, -20), (-10, -20)],
            [(-20, -20), (-20, -10), (-20, 0), (-20, 10), (-20, 20)],
            [(-10, 20), (0, 10), (10, 10), (10, 0), (10, -10)],
            [(0, -10), (-10, -10), (-10, 0), (-10, 10), (0, 0)]
        ]
        
        self.CAPTURE_TIMEOUT = 3

    def __set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def __set_last_capture_time(self, last_capture_time):
        self.last_capture_time = last_capture_time

    def __set_captured_images(self, captured_images):
        self.captured_images = captured_images

    def __calculate_head_pose(self, landmarks):
        nose = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        left_eye = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        right_eye = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
        
        eye_vector = right_eye - left_eye
        nose_vector = nose - (left_eye + eye_vector / 2)

        angle_x = np.degrees(np.arctan2(nose_vector[0], nose_vector[2])) - self.offset_x
        angle_y = np.degrees(np.arctan2(nose_vector[1], nose_vector[2])) - self.offset_y

        if angle_x < 0:
            angle_x = -180 - angle_x
        else:
            angle_x = 180 - angle_x

        return angle_x, angle_y

    def set_user_data(self, user_data):
        self.user_data = user_data
        if user_data is None:
            self.__set_offset(0, 0)
            self.__set_last_capture_time(time.time())
            self.__set_captured_images([])
            return
        self.__set_offset(user_data['offset_x'], user_data['offset_y'])
        self.__set_last_capture_time(user_data['last_capture_time'])
        self.__set_captured_images(user_data['captured_images'])

    def get_user_data(self):
        return {
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'last_capture_time': self.last_capture_time,
            'captured_images': self.captured_images
        }

    def get_next_direction(self):
        flat_directions = [(row_idx, idx) for row_idx, row in enumerate(self.directions) for idx, _ in enumerate(row)]
        for (row_idx, idx) in flat_directions:
            target_direction = f"Direction_{row_idx}_{idx}"
            if f"face_{target_direction}.jpg" not in self.captured_images:
                return self.directions[row_idx][idx], target_direction
        return None, None

    def capture(self, frame):
        user_hint = ""

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            angle_x, angle_y = self.__calculate_head_pose(landmarks)
        else:
            return False, "Wajah tidak terdeteksi"

        if self.user_data is None:
            self.__set_offset(angle_x, angle_y)

        if time.time() - self.last_capture_time > self.CAPTURE_TIMEOUT:
            next_direction, target_direction = self.get_next_direction()
            if next_direction is not None:
                horizontal_dir = "kanan" if next_direction[1] > 0 else "kiri" if next_direction[1] < 0 else ""
                vertical_dir = "atas" if next_direction[0] > 0 else "bawah" if next_direction[0] < 0 else ""

                user_hint = f"Perhatikan kamera dan arahkan wajah ke arah {horizontal_dir} {vertical_dir}"
                
            last_capture_time = time.time()

        for row_idx, row in enumerate(self.directions):
            for idx, (target_x, target_y) in enumerate(row):
                target_direction = f"[{target_x}, {target_y}]"
                if target_direction in self.captured_images:
                    continue

                if abs(angle_x - target_x) < 5 and abs(angle_y - target_y) < 10:
                    self.captured_images.append(target_direction)
                    return True, target_direction
        
        print(angle_x, angle_y)

        return False, user_hint
                    

