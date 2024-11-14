import argparse
import copy
import csv
import itertools
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp
import numpy as np

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc

#reference: pranayrishi

def get_arguments():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--device", type=int, default=0)
    argument_parser.add_argument("--width", help='cap width', type=int, default=960)
    argument_parser.add_argument("--height", help='cap height', type=int, default=540)

    argument_parser.add_argument('--use_static_image_mode', action='store_true')
    argument_parser.add_argument("--min_detection_confidence",
                                  help='min_detection_confidence',
                                  type=float,
                                  default=0.7)
    argument_parser.add_argument("--min_tracking_confidence",
                                  help='min_tracking_confidence',
                                  type=int,
                                  default=0.5)

    arguments = argument_parser.parse_args()

    return arguments

def main():
  
    arguments = get_arguments()

    video_device = arguments.device
    video_width = arguments.width
    video_height = arguments.height

    static_image_mode = arguments.use_static_image_mode
    detection_confidence = arguments.min_detection_confidence
    tracking_confidence = arguments.min_tracking_confidence

    enable_bounding_rect = True

    
    video_capture = cv.VideoCapture(video_device)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, video_width)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, video_height)

   
    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=2,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    keypoint_classifier_model = KeyPointClassifier()

    point_history_classifier_model = PointHistoryClassifier()

    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as label_file:
        keypoint_labels = csv.reader(label_file)
        keypoint_labels = [row[0] for row in keypoint_labels]
    with open(
           'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as label_file:
        point_history_labels = csv.reader(label_file)
        point_history_labels = [row[0] for row in point_history_labels]

   
    frame_rate_calculator = CvFpsCalc(buffer_len=10)

   
    history_length = 16
    point_history_deque = deque(maxlen=history_length)

    
    gesture_history_deque = deque(maxlen=history_length)

    current_mode = 0

    while True:
        fps_value = frame_rate_calculator.get()

        
        key_input = cv.waitKey(10)
        if key_input == 27:  
            break
        number, current_mode = select_mode(key_input, current_mode)

        
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = cv.flip(frame, 1) 
        debug_frame = copy.deepcopy(frame)

       
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame.flags.writeable = False
        results = hands_model.process(frame)
        frame.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
              
                bounding_rect = calc_bounding_rect(debug_frame, hand_landmarks)
               
                landmark_list = calc_landmark_list(debug_frame, hand_landmarks)

                
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_frame, point_history_deque)
              
                log_to_csv(number, current_mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier_model(pre_processed_landmark_list)
                if hand_sign_id == 2: 
                    point_history_deque.append(landmark_list[8])
                else:
                    point_history_deque.append([0, 0])

             
                finger_gesture_id = 0
                point_history_length = len(pre_processed_point_history_list)
                if point_history_length == (history_length * 2):
                    finger_gesture_id = point_history_classifier_model(
                        pre_processed_point_history_list)

                gesture_history_deque.append(finger_gesture_id)
                most_common_gesture_id = Counter(
                    gesture_history_deque).most_common()

               
                debug_frame = draw_bounding_rect(enable_bounding_rect, debug_frame, bounding_rect)
                debug_frame = draw_landmarks (debug_frame, landmark_list)
                debug_frame = draw_info_text(
                    debug_frame,
                    bounding_rect,
                    handedness,
                    keypoint_labels[hand_sign_id],
                    point_history_labels[most_common_gesture_id[0][0]],
                )
        else:
            point_history_deque.append([0, 0])

        debug_frame = draw_point_history(debug_frame, point_history_deque)
        debug_frame = draw_info(debug_frame, fps_value, current_mode, number)

       
        cv.imshow('Here is gesture recognition!', debug_frame)

    video_capture.release()
    cv.destroyAllWindows()

def select_mode(key_input, current_mode):
    number = -1
    if 48 <= key_input <= 57: 
        number = key_input - 48
    if key_input == 110:  
        current_mode = 0
    if key_input == 107: 
        current_mode = 1
    if key_input == 104: 
        current_mode = 2
    return number, current_mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_points = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_points.append([landmark_x, landmark_y])

    return landmark_points

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_value(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_value, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def log_to_csv(number, current_mode, landmark_list, point_history_list):
    if current_mode == 0:
        pass
    elif current_mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([number, *landmark_list])
    elif current_mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([number, *point_history_list])
    else:
        print("Invalid mode or number range")
    
    return

def draw_landmarks(image, landmark_points):
    if len(landmark_points) > 0:
        # Thumb
        cv.line(image, tuple(landmark_points[2]), tuple(landmark_points[3]),
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
        cv.line(image, tuple(landmark_points[0]), tuple(landmark_points [1]),
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

    # Key Points
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

def draw_bounding_rect(enable_brect, image, bounding_rect):
    if enable_brect:
        # Outer rectangle
        cv.rectangle(image, (bounding_rect[0], bounding_rect[1]), (bounding_rect[2], bounding_rect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info_text(image, bounding_rect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (bounding_rect[0], bounding_rect[1]), ( bounding_rect[2], bounding_rect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (bounding_rect[0] + 5, bounding_rect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_info(image, fps_value, current_mode, number):
    cv.putText(image, "FPS:" + str(fps_value), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps_value), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= current_mode <= 2:
        cv.putText(image, "MODE:" + mode_string[current_mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
