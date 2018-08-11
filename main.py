from flask import Flask, render_template, Response, request
from flask_sqlalchemy import SQLAlchemy
import json
import numpy as np
import face_recognition
import cv2
import os
import sys
import ast

import threading
import datetime
# ---------------------------------------------------
# Getting known faces
"""
with open('data.json') as f:
    data = json.load(f)

known_face_encodings = data["known_face_encodings"]

for i in range(len(known_face_encodings)):
    known_face_encodings[i] = np.asarray(known_face_encodings[i])
    known_face_names = data["known_face_names"]
"""
known_face_encodings = []
known_face_names = []

with open('python_config.json') as f:
    config = json.load(f)

host_name = config["localhost"]
use_blink = int(config["use_blink"])
number_of_accept_frames = int(config["number_of_detections_to_accept"])
eye_thresh = float(config["eye_threshold"])
accept_distance = float(config["face_rec_distance"])
ip_camera = config["ipcamera"]

# ---------------------------------------------------------------------------------------------
#Configuration of Main Pie
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/test.db'
db = SQLAlchemy(app)

class Face(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(80), unique = True, nullable = False)
    encoding = db.Column(db.String, unique = True, nullable = False)
    def __repr__(self):
        return '<Face %r>' % self.name

class Entering(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    entered_person = db.Column(db.String, unique = False, nullable = False)
    entered_time = db.Column(db.String, unique = False, nullable = False)
    def __repr__(self):
        return '<Entering %r>' % self.name


try:
    for face in Face.query.all():
        known_face_names.append(face.name)
        known_face_encodings.append(ast.literal_eval(face.encoding))
except:
    print("No such table")


#------------------------------------------------------------------------------------------------
#Opening Database
#with open("pq://postgres:postgres@" + host_name + ":5432/db") as f
#    db = f



#-------------------------------------------------------------------------------------------------
# for blink detection
def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.subtract(eye[1],eye[5]))
    B = np.linalg.norm(np.subtract(eye[2],eye[4]))
    C = np.linalg.norm(np.subtract(eye[0],eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear


def are_eyes_closed(face_landmark):
    leftEye = np.array(face_landmark['left_eye'])
    leftEAR = eye_aspect_ratio(face_landmark['left_eye'])
    # leftEyeHull = cv2.convexHull(leftEye)
    rightEye = np.array(face_landmark['right_eye'])
    rightEAR = eye_aspect_ratio(face_landmark['right_eye'])
    # rightEyeHull = cv2.convexHull(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    return ear < eye_thresh
#----------------------------------------------------------------------------------------------------

def face_rec(frame):
    face_locations = []
    face_encodings = []
    face_names = []
    face_landmarks = []
    rgb_small_frame = frame

    # small_frame = cv2.resize(frame, (0, 0),fx=1, fy=1)
    # rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    if use_blink:
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)
    # face recognition
    for face_encoding in face_encodings:

        distances = face_recognition.face_distance(face_encoding, known_face_encodings)

        name = "Unknown"

        mini = accept_distance
        pos = -1
        i = 0
        for distance in distances:
            if (distance < mini):
                mini = distance
                pos = i
            i += 1

        if pos > -1:
            name = known_face_names[pos]

        face_names.append(name)

    idx = 0

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Eyes
        eyes = "-"
        if use_blink:
            eyes = "Open"
            if are_eyes_closed(face_landmarks[idx]):
                eyes = "Closed"
                idx += 1

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + " " + eyes, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame, face_names
"""
def face_rec_batch(frame, face_locations, known_face_encodings, known_face_names):
    face_encodings = []
    face_names = []
    rgb_small_frame = frame

    # small_frame = cv2.resize(frame, (0, 0),fx=1, fy=1)
    # rgb_small_frame = small_frame[:, :, ::-1]
    #face_landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    # face recognition
    for face_encoding in face_encodings:

        distances = face_recognition.face_distance(face_encoding, known_face_encodings)

        name = "Unknown"

        mini = 0.5
        pos = -1
        i = 0
        for distance in distances:
            if (distance < mini):
                mini = distance
                pos = i
            i += 1

        if pos > -1:
            name = known_face_names[pos]

        face_names.append(name)

    idx = 0

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Eyes
        #eyes = "Open"
        #if are_eyes_closed(face_landmarks[idx]):
        #    eyes = "Closed"
        #idx += 1

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name , (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame
"""
prev_image = None
# --------------------------------------------------------------------------------
class VideoCamera(object):
    def __init__(self, id = "None"):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        if(id == "None"):
            self.video = cv2.VideoCapture(ip_camera)
        else:
            self.video = cv2.VideoCapture(int(id))
        self.number_of_detected_frames = dict()
        self.clear_array()
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def clear_array(self):
        threading.Timer(10.0, self.clear_array).start()
        self.number_of_detected_frames = dict()
        print("cleared")


    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        """
        self.frames.append(image)
        if len(self.frames) == 7:
            self.output_frames = []
            self.frame_count = 0
            batch_of_face_locations = face_recognition.batch_face_locations(self.frames,  number_of_times_to_upsample=1, batch_size = 7)
            
            for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                self.output_frames.append(face_rec_batch(self.frames[frame_number_in_batch], face_locations, known_face_encodings, known_face_names))
                
            self.frames = []

        if self.frame_count < len(self.output_frames):
            image = self.output_frames[self.frame_count]
        self.frame_count += 1
        """
        font = cv2.FONT_HERSHEY_DUPLEX
        global prev_image
        if(success):
            image, face_names = face_rec(image)
            prev_image = image
            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
        else:
            image = prev_image
            face_names = []

        cv2.putText(image, str(self.number_of_detected_frames), (20, 20), font, 1.0, (255, 255, 255), 1)

        for x in face_names:
            nodf = self.number_of_detected_frames
            nodf[x] = nodf.get(x, 0) + 1
            if nodf[x] == number_of_accept_frames:
                if(x != "Unknown"):
                    print(x)
                    cur_person = Entering(entered_person = x, entered_time = str(datetime.datetime.now()))
                    db.session.add(cur_person)
                    db.session.commit()
        

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
#------------------------------------------------------





@app.route('/')
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed/<id>')
def video_feed(id):
    return Response(gen(VideoCamera(id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_enterings')
def view_enterings():
    Entering_data = []
    for x in Entering.query.all():
        Name = x.entered_person
        Entered_time = x.entered_time
        Entering_data.append(Name + " " + Entered_time)
    return str(Entering_data)

@app.route('/clear_db')
def clear_db():
    Face.query.delete()
    Entering.query.delete()
    db.session.commit()
    return "done"


@app.route('/encode_faces')
def encode_faces():
    global known_face_encodings
    global known_face_names
    db.create_all()
    path = './Faces'
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(path):
        face = face_recognition.load_image_file(path + '/' + filename)
        face_locations = face_recognition.face_locations(face)
        face_encoding = face_recognition.face_encodings(face, face_locations)[0]
        known_face_encodings.append(face_encoding.tolist())
        face_name = filename.split('.')[0]
        face_code = {}
        face_code = str(face_encoding.tolist())
        known_face_names.append(filename.split('.')[0])
        cur_face = Face(name = face_name, encoding = face_code)
        old_face = Face.query.filter_by(name = face_name).first()
        if old_face is not None:   
            db.session.delete(old_face)
            db.session.commit()
        
        db.session.add(cur_face)
        db.session.commit()
    
    return 'Hello'


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == '__main__':
    app.run(host=host_name, debug=False)