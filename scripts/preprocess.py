import os
import face_recognition
import pickle

def get_face_encodings():
    base_path = "AMS/AMS/labeled_images/"
    roll_no_to_face_encoding = {}

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
            roll_no_to_face_encoding[student_roll_no] = face_recognition.face_encodings(image)[0]

    with open('face_encodings.dat', 'wb') as f:
        pickle.dump(roll_no_to_face_encoding, f)

        
   

