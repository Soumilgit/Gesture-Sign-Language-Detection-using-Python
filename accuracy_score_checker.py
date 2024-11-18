'''
Additionally, it supports LOGGING KEYPOINTS and GESTURE HISTORIES for training CLASSIFIERS.
'''
import argparse
import copy
import csv
from collections import Counter, deque
import cv2 as cv
import mediapipe as mp
import numpy as np

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc

# Argument Parser Setup
def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_device", type=int, default=0)
    parser.add_argument("--video_width", help='Video frame width', type=int, default=960)
    parser.add_argument("--video_height", help='Video frame height', type=int, default=540)

    parser.add_argument('--enable_static_image_mode', action='store_true')
    parser.add_argument("--minimum_detection_confidence",
                        help='Minimum detection confidence', type=float, default=0.7)
    parser.add_argument("--minimum_tracking_confidence",
                        help='Minimum tracking confidence', type=int, default=0.5)

    return parser.parse_args()

# Main Function
def main():
    args = get_arguments()

    input_device = args.input_device
    frame_width = args.video_width
    frame_height = args.video_height

    enable_static_image_mode = args.enable_static_image_mode
    min_detection_confidence = args.minimum_detection_confidence
    min_tracking_confidence = args.minimum_tracking_confidence

    draw_bounding_box_flag = True

    # Video capture setup
    video_capture = cv.VideoCapture(input_device)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Mediapipe Hand Detector setup
    mp_hands = mp.solutions.hands
    hand_detector = mp_hands.Hands(
        static_image_mode=enable_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Load Models
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Load Labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as file:
        keypoint_labels = csv.reader(file)
        keypoint_labels = [row[0] for row in keypoint_labels]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as file:
        point_history_labels = csv.reader(file)
        point_history_labels = [row[0] for row in point_history_labels]

    # Miscellaneous Setup
    fps_calculator = CvFpsCalc(buffer_len=10)
    max_history_length = 16
    point_history = deque(maxlen=max_history_length)
    gesture_history = deque(maxlen=max_history_length)

    mode = 0

    while True:
        frame_rate = fps_calculator.get()
        key_input = cv.waitKey(10)
        if key_input == 27:  # ESC key
            break
        number_input, mode = select_mode(key_input, mode)

        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        debug_frame = copy.deepcopy(frame)

        # Convert the image to RGB
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hand_detector.process(frame)
        frame.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                bounding_rect = calculate_bounding_box(debug_frame, hand_landmarks)
                landmark_list = calculate_landmarks(debug_frame, hand_landmarks)

                preprocessed_landmarks = preprocess_landmarks(landmark_list)
                preprocessed_history = preprocess_point_history(debug_frame, point_history)

                save_to_csv(number_input, mode, preprocessed_landmarks, preprocessed_history)

                hand_sign_id = keypoint_classifier(preprocessed_landmarks)
                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])  # Index finger tip
                else:
                    point_history.append([0, 0])

                gesture_id = 0
                if len(preprocessed_history) == (max_history_length * 2):
                    gesture_id = point_history_classifier(preprocessed_history)

                gesture_history.append(gesture_id)
                most_common_gesture = Counter(gesture_history).most_common()

                debug_frame = draw_bounding_box(draw_bounding_box_flag, debug_frame, bounding_rect)
                debug_frame = draw_landmarks(debug_frame, landmark_list)
                debug_frame = draw_info_text(
                    debug_frame,
                    bounding_rect,
                    hand_type,
                    keypoint_labels[hand_sign_id],
                    point_history_labels[most_common_gesture[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_frame = draw_point_history(debug_frame, point_history)
        debug_frame = draw_info(debug_frame, frame_rate, mode, number_input)

        cv.imshow('Gesture Recognition System', debug_frame)

    video_capture.release()
    cv.destroyAllWindows()

def select_mode(key, current_mode):
    if 48 <= key <= 57:  # Numbers 0-9
        return key - 48, current_mode
    if key == 110:  # 'n' key
        return -1, (current_mode + 1) % 3
    return -1, current_mode

# Calculate Bounding Box
def calculate_bounding_box(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    x_coordinates = [landmark.x * image_width for landmark in landmarks.landmark]
    y_coordinates = [landmark.y * image_height for landmark in landmarks.landmark]

    x_min = max(0, int(min(x_coordinates) - 10))
    y_min = max(0, int(min(y_coordinates) - 10))
    x_max = min(image_width, int(max(x_coordinates) + 10))
    y_max = min(image_height, int(max(y_coordinates) + 10))

    return [x_min, y_min, x_max, y_max]

# Calculate Landmarks
def calculate_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []
    for landmark in landmarks.landmark:
        landmark_x = int(landmark.x * image_width)
        landmark_y = int(landmark.y * image_height)
        landmark_list.append([landmark_x, landmark_y])
    return landmark_list

# Preprocess Landmarks
def preprocess_landmarks(landmarks):
    base_x, base_y = landmarks[0]
    normalized_landmarks = []
    for x, y in landmarks:
        normalized_landmarks.append([(x - base_x) / base_x, (y - base_y) / base_y])
    return np.array(normalized_landmarks).flatten()

# Preprocess Point History
def preprocess_point_history(image, history):
    image_width, image_height = image.shape[1], image.shape[0]
    preprocessed_history = []
    for point in history:
        preprocessed_history.append([point[0] / image_width, point[1] / image_height])
    return np.array(preprocessed_history).flatten()

# Save to CSV
def save_to_csv(number, mode, landmarks, history):
    if number < 0:
        return
    mode_path = 'model/keypoint_classifier/keypoint_classifier_data.csv' if mode == 0 else 'model/point_history_classifier/point_history_classifier_data.csv'
    data = landmarks if mode == 0 else history
    with open(mode_path, 'a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([number, *data])

# Drawing Functions
def draw_bounding_box(draw_flag, image, bounding_box):
    if not draw_flag:
        return image
    cv.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
    return image

def draw_landmarks(image, landmark_points):
    if len(landmark_points) > 0:
        # Thumb
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[ 3]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[3]), tuple(landmark_points[4]),
                (0, 128, 0), 2)

        # Index finger
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[6]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[6]), tuple(landmark_points[7]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[7]), tuple(landmark_points[8]),
                (0, 128, 0), 2)

        # Middle finger
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[10]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[10]), tuple(landmark_points[11]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[11]), tuple(landmark_points[12]),
                (0, 128, 0), 2)

        # Ring finger
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[14]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[14]), tuple(landmark_points[15]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[15]), tuple(landmark_points[16]),
                (0, 128, 0), 2)

        # Little finger
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[18]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[18]), tuple(landmark_points[19]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[19]), tuple(landmark_points[20]),
                (0, 128, 0), 2)

        # Palm
        cv.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[0]), tuple(landmark_points[1]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[1]), tuple(landmark_points[2]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[5]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[5]), tuple(landmark_points[9]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[9]), tuple(landmark_points[13]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[13]), tuple(landmark_points[17]),
                (0, 128, 0), 2)
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]),
                (255, 165, 0), 6)
        cv.line(image, tuple(landmark_points[17]), tuple(landmark_points[0]),
                (0, 128, 0), 2)

    for index, landmark in enumerate(landmark_points):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif index == 1: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif index == 3: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif index == 4: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
    return image

def draw_bounding_box(draw_flag, image, bounding_box):
    if not draw_flag:
        return image
    cv.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
    return image

def draw_landmarks(image, landmarks):
    for index, landmark in enumerate(landmarks):
        cv.circle(image, tuple(landmark), 5, (0, 255, 255), -1)
    return image

def draw_info_text(image, bounding_box, hand_type, hand_label, gesture_label):
    info_text = f"{hand_type}: {hand_label} ({gesture_label})"
    cv.putText(image, info_text, (bounding_box[0], bounding_box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def draw_point_history(image, history):
    for point in history:
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, tuple(point), 5, (255, 0, 0), -1)
    return image

def draw_info(image, fps, mode, number_input):
    mode_info = ['Logging Keypoints', 'Logging History', 'Inference Mode']
    cv.putText(image, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(image, f"Mode: {mode_info[mode]}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if number_input >= 0:
        cv.putText(image, f"Number: {number_input}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

if __name__ == '__main__':
    main()

