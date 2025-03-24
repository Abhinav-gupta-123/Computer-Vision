import cv2
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(model_path, input_size=(320, 320), score_threshold=0.8, nms_threshold=0.3):
    """
    Load the YuNet face detection model.
    """
    try:
        model = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            nms_threshold=nms_threshold
        )
        model.setScoreThreshold(score_threshold)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

#function for detecting face
def detect_faces(model, frame):
    h, w, _ = frame.shape
    model.setInputSize((w, h))
    _, faces = model.detect(frame)
    return faces

def draw_bounding_boxes(frame, faces):
    """
    Draw bounding boxes around detected faces.
    """
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            confidence = face[4]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def main(model_path, video_source=0):
    """
    Main function to run face detection on a video source.
    """
    # Load the model
    model = load_model(model_path)
    if model is None:
        return

    # Open the video source
    source = cv2.VideoCapture(video_source)
    source.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    source.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not source.isOpened():
        logging.error("Failed to open video source.")
        return

    logging.info("Starting face detection...")
    while cv2.waitKey(1) != 27:  # Press ESC to exit
        ret, frame = source.read()
        if not ret:
            logging.warning("Failed to capture frame.")
            break

        # Detect faces
        faces = detect_faces(model, frame)

        # Draw bounding boxes
        frame = draw_bounding_boxes(frame, faces)

        # Display the result
        cv2.imshow("YuNet Face Detection", frame)

    # Release resources
    source.release()
    cv2.destroyAllWindows()
    logging.info("Face detection stopped.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face Detection using YuNet")
    parser.add_argument("--model", type=str, default=r"C:\Users\abhin\Desktop\computer_vision\face_detection_using_yolov5\face_detection_yunet_2023mar.onnx",
                        help="Path to the YuNet ONNX model file (default: %(default)s)")
    parser.add_argument("--source", type=int, default=0,
                        help="Video source (0 for webcam or file path, default: %(default)s)")
    args = parser.parse_args()

    # Run the main function
    main(model_path=args.model, video_source=args.source)