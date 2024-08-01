import cv2

def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to save a picture or 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite('original_frame.jpg', frame)
            print("Image saved as 'captured_image.jpg'.")
        
        elif key == ord('q'):
            print("Quitting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image_from_webcam()
