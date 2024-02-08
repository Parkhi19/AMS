import os
import cv2
import face_recognition
import sys

def get_face_encodings():
    base_path = "AMS/labeled_images/"
    known_face_encodings = []
    known_student_roll_no = []

    for student_folder in os.listdir(base_path):
        student_roll_no = student_folder
        student_folder = base_path + student_folder
        for image in os.listdir(student_folder):
            image = student_folder + "/" + image
            # print(image)
            image = face_recognition.load_image_file(image)
            if(len(face_recognition.face_encodings(image)) == 0):
                print("No face found in the image")
                continue
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_student_roll_no.append(student_roll_no)
    return known_face_encodings, known_student_roll_no


   

