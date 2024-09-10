import cv2
import numpy as np
import threading
import time
from lib.faceIdentifier import FaceIdentifier
import uuid
import os

os.system('cls' if os.name == 'nt' else 'clear')

def register():
    camera = cv2.VideoCapture(0)
    face_identifier = FaceIdentifier()
    countdown = 5

    # Registering user images by taking 5 pictures with interval of 1 second
    name = input('Enter your name: ')
    frames = []
    id = str(uuid.uuid4())

    def overlay_timer(frame, countdown):
        """Overlay the timer and black layer on the frame."""
        black_layer = np.zeros_like(frame, dtype=np.uint8)
        black_layer[:] = (0, 0, 0)
        frame_with_overlay = cv2.addWeighted(black_layer, 0.5, frame, 0.5, 0)

        cv2.putText(frame_with_overlay, f"Registering in {countdown}...", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        
        return frame_with_overlay

    def update_timer():
        nonlocal countdown
        while countdown > 0:
            countdown -= 1
            time.sleep(1)

    timer_thread = threading.Thread(target=update_timer)
    timer_thread.start()

    while countdown > 0:
        success, frame = camera.read()
        if not success:
            print('Error reading the camera')
            return

        frame_with_overlay = overlay_timer(frame, countdown)
        cv2.imshow('frame', frame_with_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for i in range(10):
        success, frame = camera.read()
        frames.append(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)

    face_identifier.register(id, name, frames)

    print('Images registered successfully. ID:', id)

    camera.release()
    cv2.destroyAllWindows()

def overlay_timer(frame, countdown):
    """Overlay the timer and black layer on the frame."""
    black_layer = np.zeros_like(frame, dtype=np.uint8)
    black_layer[:] = (0, 0, 0)
    frame_with_overlay = cv2.addWeighted(black_layer, 0.5, frame, 0.5, 0)

    cv2.putText(frame_with_overlay, f"Identifying in {countdown}...", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    
    return frame_with_overlay

def identify(id):
    camera = cv2.VideoCapture(0)
    face_identifier = FaceIdentifier()
    countdown = 5
    
    def update_timer():
        nonlocal countdown
        while countdown > 0:
            countdown -= 1
            time.sleep(1)
    
    timer_thread = threading.Thread(target=update_timer)
    timer_thread.start()

    while countdown > 0:
        success, frame = camera.read()
        if not success:
            print('Error reading the camera')
            return

        frame_with_overlay = overlay_timer(frame, countdown)
        cv2.imshow('frame', frame_with_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the countdown, capture the frame for identification
    success, frame = camera.read()
    cv2.imshow('frame', frame)
    if success:
        name = face_identifier.identify(id, frame)
        if name == 'Unknown':
            print('Unknown person')
        else:
            print('\033[44m\033[37m[*] \033[30m\033[107m Welcome', name, '!!! \033[0m')

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while True:
        print('1. Register\n2. Identify\n3. Exit')
        option = input('Select an option: ')
        os.system('cls' if os.name == 'nt' else 'clear')

        if option == '1':
            register()
        elif option == '2':
            id = input('Enter your ID: ')
            if os.path.exists(f'./images/{id}'):
                identify(id)
            else:
                print('Invalid ID')
        elif option == '3':
            break
        else:
            print('Invalid option')
