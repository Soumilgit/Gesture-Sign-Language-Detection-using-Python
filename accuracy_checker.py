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

#Reference: pranayrishi

def get_the_arguments():
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("--device", type=int, default=0)
    arg_parse.add_argument("--width", help='cap width', type=int, default=960)
    arg_parse.add_argument("--height", help='cap height', type=int, default=540)

    arg_parse.add_argument('--use_static_img_mode', action='store_true')
    arg_parse.add_argument("--min_detect_conf",
                                  help='min_detect_conf',
                                  type=float,
                                  default=0.7)
    arg_parse.add_argument("--min_track_conf",
                                  help='min_track_conf',
                                  type=int,
                                  default=0.5)

    arguments = arg_parse.parse_args()

    return arguments

def main():
  
    arguments = get_the_arguments()

    vid_d = arguments.device
    vid_w = arguments.width
    vid_h = arguments.height

    static_img_mode = arguments.use_static_img_mode
    detect_conf = arguments.min_detect_conf
    track_conf = arguments.min_track_conf

    enable_bounding_rect = True

    vid_capt = cv.VideoCapture(vid_d)
    vid_capt.set(cv.CAP_PROP_FRAME_WIDTH, vid_w)
    vid_capt.set(cv.CAP_PROP_FRAME_HEIGHT, vid_h)

    mp_h = mp.solutions.hands
    h_model = mp_h.Hands(
        static_image_mode=static_img_mode,
        max_num_hands=2,
        min_detection_confidence=detect_conf,
        min_tracking_confidence=track_conf,
    )

    keypt_classf_model = KeyPointClassifier()
    pt_history_classf_model = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as label_file:
        keypt_labels = csv.reader(label_file)
        keypt_labels = [row[0] for row in keypt_labels]
    with open(
           'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as label_file:
        pt_history_labels = csv.reader(label_file)
        pt_history_labels = [row[0] for row in pt_history_labels]

    frame_rate_calc = CvFpsCalc(buffer_len=10)
    history_l = 16
    pt_history_deq = deque(maxlen=history_l)
    gesture_history_deq = deque(maxlen=history_l)

    curr_mode = 0

    while True:
        fps_val = frame_rate_calc.get()
        key_inp = cv.waitKey(10)
        if key_inp == 27:  
            break
        num, curr_mode = select_mode(key_inp, curr_mode)

        ret, frame = vid_capt.read()
        if not ret:
            break
        frame = cv.flip(frame, 1) 
        debug_frame = copy.deepcopy(frame)

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        parinam = h_model.process(frame)
        frame.flags.writeable = True

        if parinam.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(parinam.multi_hand_landmarks,
                                              parinam.multi_handedness):
                bounding_rect = calc_bounding_rect(debug_frame, hand_landmarks)
                landmark_list = calc_landmark_list(debug_frame, hand_landmarks)

                pre_proc_landmark_list = pre_proc_landmark(landmark_list)
                pre_proc_pt_history_list = pre_proc_pt_history(
                    debug_frame, pt_history_deq)
              
                log_to_csv(num, curr_mode, pre_proc_landmark_list,
                            pre_proc_pt_history_list)

                hand_sign_id = keypt_classf_model(pre_proc_landmark_list)
                if hand_sign_id == 2: 
                    pt_history_deq.append(landmark_list[8])
                else:
                    pt_history_deq.append([0, 0])
                
                

                finger_gesture_id = 0
                point_history_length = len(pre_proc_pt_history_list)
                if point_history_length == (history_l * 2):
                    finger_gesture_id = pt_history_classf_model(
                        pre_proc_pt_history_list)

                gesture_history_deq.append(finger_gesture_id)
                most_common_gesture_id = Counter(
                    gesture_history_deq).most_common()

                debug_frame = draw_bound_rect(enable_bounding_rect, debug_frame, bounding_rect)
                debug_frame = draw_landmarks(debug_frame, landmark_list)
                debug_frame = draw_information_txt(
                    debug_frame,
                    bounding_rect ,
                    handedness,
                    keypt_labels[hand_sign_id],
                    pt_history_labels[most_common_gesture_id[0][0]],
                )
        else:
            pt_history_deq.append([0, 0])

        debug_frame = draw_pt_history(debug_frame, pt_history_deq)
        debug_frame = draw_information(debug_frame,  fps_val, curr_mode, num)

        cv.imshow('Here is a gesture recognition tool!', debug_frame)

    vid_capt.release()
    cv.destroyAllWindows()

def select_mode(key_inp, curr_mode):
    num = -1
    if 48 <= key_inp <= 57: 
        num = key_inp - 48
    if key_inp == 110:  
        curr_mode = 0
    if key_inp == 107: 
        curr_mode = 1
    if key_inp == 104: 
        curr_mode = 2
    return num, curr_mode

def calc_bounding_rect(img, landmarks):
    img_w, img_h = img.shape[1], img.shape[0]
    landmark_arr = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_w), img_w - 1)
        landmark_y = min(int(landmark.y * img_h), img_h - 1)
        landmark_pt = [np.array((landmark_x, landmark_y))]
        landmark_arr = np.append(landmark_arr, landmark_pt, axis=0)

    x, y, w, h = cv.boundingRect(landmark_arr)
    return [x, y, x + w, y + h]

def calc_landmark_list(img, landmarks):
    img_w, img_h = img.shape[1], img.shape[0]
    landmark_pts = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * img_w), img_w - 1)
        landmark_y = min(int(landmark.y * img_h), img_h - 1)
        landmark_pts.append([landmark_x, landmark_y])

    return landmark_pts

def pre_proc_landmark(landmark_lst):
    temporary_landmark_list = copy.deepcopy(landmark_lst)
    b_x, b_y = 0, 0
    for ind, landmark_pt in enumerate(temporary_landmark_list):
        if ind == 0:
            b_x, b_y = landmark_pt[0], landmark_pt[1]
        temporary_landmark_list[ind][0] = temporary_landmark_list[ind][0] - b_x
        temporary_landmark_list[ind][1] = temporary_landmark_list[ind][1] - b_y

    temporary_landmark_list = list(itertools.chain.from_iterable(temporary_landmark_list))
    maximum_val = max(list(map(abs, temporary_landmark_list)))

    def normalize_val(n):
        return n / maximum_val

    temporary_landmark_list = list(map(normalize_val, temporary_landmark_list))
    return temporary_landmark_list

def pre_proc_pt_history(img, pt_history):
    img_w, img_h = img.shape[1], img.shape[0]
    temporary_pt_history = copy.deepcopy(pt_history)
    b_x, b_y = 0, 0
    for ind, pt in enumerate(temporary_pt_history):
        if ind == 0:
            b_x, b_y = pt[0], pt[1]
        temporary_pt_history[ind][0] = (temporary_pt_history[ind][0] - b_x) / img_w
        temporary_pt_history[ind][1] = (temporary_pt_history[ind][1] - b_y) / img_h

    temporary_pt_history = list(itertools.chain.from_iterable(temporary_pt_history))
    return temporary_pt_history

def log_to_csv(num, curr_mode, landmark_lst, pt_history_list):
    if curr_mode == 0:
        pass
    elif curr_mode == 1 and (0 <= num <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([num, *landmark_lst])
    elif curr_mode == 2 and (0 <= num <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as file:
            writer = csv.writer(file)
            writer.writerow([num, *pt_history_list])
    else:
        print("Invalid mode or number range")
    
    return

def draw_landmarks(img, landmark_pts):
    if len(landmark_pts) > 0:
        # Thumb
        cv.line(img, tuple(landmark_pts[2]), tuple(landmark_pts[ 3]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[2]), tuple(landmark_pts[3]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[3]), tuple(landmark_pts[4]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[3]), tuple(landmark_pts[4]),
                (0, 128, 0), 2)

        # Index finger
        cv.line(img, tuple(landmark_pts[5]), tuple(landmark_pts[6]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[5]), tuple(landmark_pts[6]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[6]), tuple(landmark_pts[7]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[6]), tuple(landmark_pts[7]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[7]), tuple(landmark_pts[8]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[7]), tuple(landmark_pts[8]),
                (0, 128, 0), 2)

        # Middle finger
        cv.line(img, tuple(landmark_pts[9]), tuple(landmark_pts[10]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[9]), tuple(landmark_pts[10]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[10]), tuple(landmark_pts[11]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[10]), tuple(landmark_pts[11]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[11]), tuple(landmark_pts[12]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[11]), tuple(landmark_pts[12]),
                (0, 128, 0), 2)

        # Ring finger
        cv.line(img, tuple(landmark_pts[13]), tuple(landmark_pts[14]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[13]), tuple(landmark_pts[14]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[14]), tuple(landmark_pts[15]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[14]), tuple(landmark_pts[15]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[15]), tuple(landmark_pts[16]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[15]), tuple(landmark_pts[16]),
                (0, 128, 0), 2)

        # Little finger
        cv.line(img, tuple(landmark_pts[17]), tuple(landmark_pts[18]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[17]), tuple(landmark_pts[18]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[18]), tuple(landmark_pts[19]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[18]), tuple(landmark_pts[19]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[19]), tuple(landmark_pts[20]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[19]), tuple(landmark_pts[20]),
                (0, 128, 0), 2)

        # Palm
        cv.line(img, tuple(landmark_pts[0]), tuple(landmark_pts[1]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[0]), tuple(landmark_pts[1]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[1]), tuple(landmark_pts[2]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[1]), tuple(landmark_pts[2]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[2]), tuple(landmark_pts[5]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[2]), tuple(landmark_pts[5]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[5]), tuple(landmark_pts[9]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[5]), tuple(landmark_pts[9]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[9]), tuple(landmark_pts[13]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[9]), tuple(landmark_pts[13]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[13]), tuple(landmark_pts[17]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[13]), tuple(landmark_pts[17]),
                (0, 128, 0), 2)
        cv.line(img, tuple(landmark_pts[17]), tuple(landmark_pts[0]),
                (255, 165, 0), 6)
        cv.line(img, tuple(landmark_pts[17]), tuple(landmark_pts[0]),
                (0, 128, 0), 2)

    for ind, landmark in enumerate(landmark_pts):
        if ind == 0:
            cv.circle(img, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(img, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif ind == 1: 
            cv.circle(img, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(img, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif ind == 2:  
            cv.circle(img, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(img, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif ind == 3: 
            cv.circle(img, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(img, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif ind == 4: 
            cv.circle(img, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(img, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
        elif ind == 5:  
            cv.circle(img, (landmark[0], landmark[1]), 5, (255, 165, 0),
                      -1)
            cv.circle(img, (landmark[0], landmark[1]), 5, (0, 128, 0), 1)
    return img

def draw_bound_rect(enable_brect, img, bound_rectangle):
    if enable_brect:
        cv.rectangle(img, (bound_rectangle[0], bound_rectangle[1]), (bound_rectangle[2], bound_rectangle[3]),
                     (0, 0, 0), 1)
    return img

def draw_information_txt(img, bound_rectangle, handedness, hand_sign_txt,
                   finger_gesture_text):
    cv.rectangle(img, (bound_rectangle[0], bound_rectangle[1]), (bound_rectangle[2], bound_rectangle[1] - 22),
                 (0, 0, 0), -1)

    information_text = handedness.classification [0].label[0:]
    if hand_sign_txt != "":
        information_text = information_text + ':' + hand_sign_txt
    cv.putText(img, information_text, (bound_rectangle[0] + 5, bound_rectangle[1] - 4),
               cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(img, "Gesture of finger->" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(img, "Gesture of finger->" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return img

def draw_pt_history(img, pt_history):
    for ind, point in enumerate(pt_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(img, (point[0], point[1]), 1 + int(ind / 2),
                      (152, 251, 152), 2)

    return img

def draw_information(img,fps_val, curr_mode, num):
    cv.putText(img, "Frames Per Second->" + str(fps_val), (10, 30), cv.FONT_HERSHEY_COMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(img, "Frames Per Second->" + str(fps_val), (10, 30), cv.FONT_HERSHEY_COMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= curr_mode <= 2:
        cv.putText(img, "Mode->" + mode_string[curr_mode - 1], (10, 90),
                   cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= num <= 9:
            cv.putText(img, "Num->" + str(num), (10, 110),
                       cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return img

if __name__ == '__main__':
    main()

