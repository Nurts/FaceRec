import os
import json
import face_recognition

def write_json(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.close()

path = './Faces'
known_face_encodings = []
known_face_names = []

for filename in os.listdir(path):
    face = face_recognition.load_image_file(path + '/' + filename)
    face_locations = face_recognition.face_locations(face, model = "cnn")
    face_encoding = face_recognition.face_encodings(face, face_locations)[0]
    known_face_encodings.append(face_encoding.tolist())
    known_face_names.append(filename.split('.')[0])

data = {}

data["known_face_encodings"] = known_face_encodings
data["known_face_names"] = known_face_names

write_json(data)
print("DONE")