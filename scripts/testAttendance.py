import face_recognition
import pickle
import os
import numpy as np

file_path = os.path.abspath('D:/Work/AMS Everything/AMS/AMS/scripts/face_encodings.dat')


with open(file_path, 'rb') as f:
    all_face_encodings = pickle.load(f)


face_names = list(all_face_encodings.keys())
face_encodings = np.array(list(all_face_encodings.values())) 

unknown_image_path = os.path.abspath('D:/Work/AMS Everything/AMS/AMS/scripts/face_image_23.jpg')

unknown_image = face_recognition.load_image_file(unknown_image_path)
unknown_face = face_recognition.face_encodings(unknown_image)

result = face_recognition.compare_faces(face_encodings, unknown_face)

names_with_result = list(zip(face_names, result))
print(names_with_result)