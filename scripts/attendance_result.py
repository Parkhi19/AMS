import face_recognition
import pickle
import os
import numpy as np

file_path = os.path.abspath('D:/Work/AMS Everything/AMS/AMS/scripts/face_encodings.dat')

with open(file_path, 'rb') as f:
    all_face_encodings = pickle.load(f)

face_names = list(all_face_encodings.keys())
face_encodings = np.array(list(all_face_encodings.values())) 

unknown_images_folder = os.path.abspath('D:/Work/AMS Everything/AMS/AMS/scripts/unknown_images')

all_results = []

for image_file in os.listdir(unknown_images_folder):
    image_path = os.path.join(unknown_images_folder, image_file)  
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    if len(unknown_face_encodings) > 0:
        results = face_recognition.compare_faces(face_encodings, unknown_face_encodings[0])
        rollno_with_result = list(zip(face_names, results))
        all_results.append((image_file, rollno_with_result))

for image_file, results in all_results:
    print(f"Results for image: {image_file}")
    print(results)
