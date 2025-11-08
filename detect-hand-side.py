import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np

# Load YOLOv7 model
def load_model(weights):
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights=weights, trust_repo=True)
    model.eval()
    return model

def main(weights):
    model = load_model(weights)

    cap = cv2.VideoCapture(0)  # webcam
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)
        detections = results.xyxy[0]  # (x1, y1, x2, y2, conf, class)

        frame_width = frame.shape[1]
        right_boundary = frame_width * (2/3)  # right third of screen

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Compute center of detected hand
            cx = (x1 + x2) // 2

            # Draw center point
            cv2.circle(frame, (cx, y1), 5, (0, 0, 255), -1)

            # Check if hand is on the right side
            if cx > right_boundary:
                cv2.putText(frame, "HAND ON RIGHT SIDE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("HAND ON RIGHT SIDE")

        # Show output
        cv2.imshow("Hand Detection (Right Side Trigger)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="yolov7.pt")
    args = parser.parse_args()

    main(args.weights)
