from ultralytics import YOLO
import cv2
import numpy as np


def fall_detection():

    model_fall = YOLO("ckpts/fall_det_1.pt")

    model_person = YOLO("yolov8n.pt")

    video_path = "fall_detc.mp4"  
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    last_logged_second = -1
    fall_events = []  # Store (frame_number, time_in_seconds)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('fall_detection_output.avi', fourcc, 30.0, (640, 480))

    fall_detected_frames = 0
    FALL_THRESHOLD = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height = frame.shape[0]
        detected_fall = False

        fall_boxes = []
        fall_results = model_fall.predict(frame)
        for result in fall_results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls].lower()
                if "fall" in label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    fall_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Fall-Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        person_results = model_person.predict(frame)
        for result in person_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                if label != 'person':
                    continue

                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                center_y = (y_min + y_max) // 2
                aspect_ratio = (y_max - y_min) / (x_max - x_min)

                person_box_color = (0, 255, 0)
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), person_box_color, 2)
                # cv2.putText(frame, "Person", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, person_box_color, 2)

                for fx1, fy1, fx2, fy2 in fall_boxes:
                    inter_x1 = max(x_min, fx1)
                    inter_y1 = max(y_min, fy1)
                    inter_x2 = min(x_max, fx2)
                    inter_y2 = min(y_max, fy2)
                    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    person_area = (x_max - x_min) * (y_max - y_min)
                    iou = inter_area / person_area

                    if iou > 0.3 or (aspect_ratio < 0.5 and center_y > frame_height * 0.7):
                        detected_fall = True
                        break

        if detected_fall:
            fall_detected_frames += 1
        else:
            fall_detected_frames = 0

        if fall_detected_frames >= FALL_THRESHOLD:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_second = int(current_frame / fps)

            if current_second != last_logged_second:
                timestamp = f"{int(current_second // 60):02d}:{int(current_second % 60):02d}"
                fall_events.append((current_frame, timestamp))
                print(f"[ALERT] Fall detected at {timestamp}")
                last_logged_second = current_second

                # cv2.putText(frame, "\u26a0 FALL ALERT TRIGGERED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)
        # cv2.imshow("Fall Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return fall_events