import face_recognition_models
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

face_recognition_models.path = ['models']

video_capture = cv2.VideoCapture(0)

my_image = face_recognition.load_image_file("C:/Users/bauss/Desktop/Open CV/Dr Moshin Jadoon.jpg")
myimg_encoding = face_recognition.face_encodings(my_image)[0]
my_image2 = face_recognition.load_image_file("C:/Users/bauss/Desktop/Open CV/ali_naqi.jpeg")
myimg2_encoding = face_recognition.face_encodings(my_image2)[0]


known_face_encodings = [myimg_encoding, myimg2_encoding]
known_face_names = ["Dr Moshin Jadoon", "Ali Naqi"]

students = known_face_names.copy()
attendance_record = {student: False for student in students}

prev_face_landmarks = None
face_distance_threshold = 0.6  # Adjust the threshold as needed
motion_threshold = 10  # Adjust the threshold as needed

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index] and face_distances[best_match_index] < face_distance_threshold:
            name = known_face_names[best_match_index]

            landmarks = face_recognition.face_landmarks(rgb_small_frame, [(top, right, bottom, left)])[0]

            if prev_face_landmarks is not None:
                flat_landmarks = [item for sublist in landmarks.values() for item in sublist]
                prev_flat_landmarks = [item for sublist in prev_face_landmarks.values() for item in sublist]
                motion = cv2.norm(np.array(flat_landmarks), np.array(prev_flat_landmarks), cv2.NORM_L2)

                if motion > motion_threshold and not attendance_record[name]:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("Naqi883.csv", "a", newline="") as csvfile:
                        lnwriter = csv.writer(csvfile)
                        lnwriter.writerow([name, current_time])
                    attendance_record[name] = True

            if name in students:
                students.remove(name)

            prev_face_landmarks = landmarks

            font = cv2.FONT_HERSHEY_COMPLEX
            bottom_left_corner_of_text = (10, 100)
            font_scale = 1.5
            font_color = (255, 0, 0)
            thickness = 3
            line_type = 2
            cv2.putText(frame, f"{name} Present", bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
