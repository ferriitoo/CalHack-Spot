from flask import Flask, render_template, request
import requests




import paramiko
import sys
import time


def execute_command_in_robot(char):

    # Configura la conexión SSH
    host = '192.168.80.3'
    port = 20022
    username = 'spot'
    password = 'Merkleb0t'

    # Crea una instancia SSHClient
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(host, port, username, password)
    print("Conexión SSH establecida")

    print("Press keys (no need to press Enter) to send to the remote script.")
    print("Press 'q' to quit.")

    # Construye el comando para ejecutar el programa en el robot con el carácter como argumento
    comando = f'python3.6 -u /home/spot/julen_yash/julen.py {char}'

    print(f'CHAR: {char}')
    print('EXECUTED COMMAND:', comando)

    # Ejecuta el comando en the robot
    stdin, stdout, stderr = ssh.exec_command(comando)

    # Wait for the command to complete
    stdout.channel.recv_exit_status()

    # Print the standard output and error
    print("STDOUT:", stdout.read().decode())
    print("STDERR:", stderr.read().decode())

    # Cierra la conexión SSH
    ssh.close()
    print("Conexión SSH cerrada")



app = Flask(__name__)



# YASH
# ----------------------------------------------------------------




import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
import time
import paramiko


# def main():
    
#     cap = cv2.VideoCapture(0)
    
#     while(True):
#         cap.release()
#         cv2.destroyAllWindows()
        
    
    
# ----------------------------------------------------------------------

import cv2
import numpy as np
from flask import Flask, render_template, Response


class VideoCamera(object):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_image = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.action_list = ["left", "right", "backward", "forward", "stop"]
        self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.isdebug = False
        self.with_authentication = False
        self.use_video_stream = False
        if not self.isdebug:
            time.sleep(1)
            self.execute_command_in_robot("l")
            time.sleep(5)
            self.execute_command_in_robot("P")
            time.sleep(5)
            self.execute_command_in_robot("f")
            time.sleep(1)
        
        self.last_entry = [time.time()]*len(self.action_list)
        self.global_actions = []
        self.buffer = 10
        resolution = (640, 480)
        if self.with_authentication:
            self.owner = cv2.imread("yash.jpg")
            self.owner = cv2.resize(self.owner, resolution)
        if self.use_video_stream:
            self.video = cv2.VideoCapture("http://10.66.31.34:8080/video")
        else:
            self.video = cv2.VideoCapture(0)
        if not (self.video.isOpened()):
            print("Could not open video device")
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cnt = 0
        self.prev_result = []
    
    def execute_command_in_robot(self, char):

        # Configura la conexión SSH
        host = '192.168.80.3'
        port = 20022
        username = 'spot'
        password = 'Merkleb0t'

        # Crea una instancia SSHClient
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(host, port, username, password)
        print("Conexión SSH establecida")

        print("Press keys (no need to press Enter) to send to the remote script.")
        print("Press 'q' to quit.")

        # Construye el comando para ejecutar el programa en el robot con el carácter como argumento
        comando = f'python3.6 -u /home/spot/julen_yash/julen.py {char}'

        print(f'CHAR: {char}')
        print('EXECUTED COMMAND:', comando)

        # Ejecuta el comando en the robot
        stdin, stdout, stderr = ssh.exec_command(comando)

        # Wait for the command to complete
        stdout.channel.recv_exit_status()

        # Print the standard output and error
        print("STDOUT:", stdout.read().decode())
        print("STDERR:", stderr.read().decode())

        # Cierra la conexión SSH
        ssh.close()
        print("Conexión SSH cerrada")
    
    def Distance_finder(self, Focal_Length, real_face_width, face_width_in_frame):
        distance = (real_face_width * Focal_Length) / face_width_in_frame
        return distance

    def face_data(self, image, CallOut, Distance_level):
        face_width = 0
        face_x, face_y = 0, 0
        face_center_x = 0
        face_center_y = 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in faces:
            line_thickness = 2
            # print(len(faces))
            GREEN = (0, 255, 0)
            LLV = int(h * 0.12)

            # print(LLV)

            # cv2.rectangle(image, (x, y), (x+w, y+h), BLACK, 1)
            cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
            cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
            cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
            cv2.line(
                image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness
            )
            cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
            cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)

            face_width = w
            face_center = []
            # Drwaing circle at the center of the face
            face_center_x = int(w / 2) + x
            face_center_y = int(h / 2) + y
            if Distance_level < 10:
                Distance_level = 10

            # cv2.circle(image, (face_center_x, face_center_y),5, (255,0,255), 3 )
            if CallOut == True:
                # cv2.line(image, (x,y), (face_center_x,face_center_y ), (155,155,155),1)
                cv2.line(image, (x, y - 11), (x + 180, y - 11), ((0, 69, 255)), 28)
                cv2.line(image, (x, y - 11), (x + 180, y - 11), ((0, 255, 255)), 20)
                cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), (GREEN), 18)

                # cv2.circle(image, (face_center_x, face_center_y),2, (255,0,255), 1 )
                # cv2.circle(image, (x, y),2, (255,0,255), 1 )

            # face_x = x
            # face_y = y

        return face_width, faces, face_center_x, face_center_y

    def calculate_dist(self, a, b):
        A = np.array([a.x, a.y]) 
        B = np.array([b.x, b.y]) 
        return np.linalg.norm(A - B)

    def calculate_angle(self, a, b, c, thresh = 0.8, use_2d = True):
        if a.visibility < thresh or b.visibility < thresh or c.visibility < thresh:
            return 0.0
        
        if use_2d:
            A = np.array([a.x, a.y]) 
            B = np.array([b.x, b.y]) 
            C = np.array([c.x, c.y])  
        else:
            A = np.array([a.x, a.y, a.z]) 
            B = np.array([b.x, b.y, b.z])  
            C = np.array([c.x, c.y, c.z])  

        AB = B - A
        BC = C - B

        dot_product = np.dot(AB, BC)

        magnitude_AB = np.linalg.norm(AB)
        magnitude_BC = np.linalg.norm(BC)

        angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_BC))

        angle_degrees = np.degrees(angle_radians)
        return angle_degrees

    def calculate_perpendicular_angle(self, a, b, thresh = 0.8, isleft = False):
        if a.visibility < thresh or b.visibility < thresh:
            if isleft:
                return 180
            return 0.0
        point1 = np.array([a.x, a.y]) 
        point2 = np.array([b.x, b.y]) 
    
        vertical_distance = np.abs(point2[1] - point1[1])
        horizontal_distance = np.abs(point2[0] - point1[0])
        angle_radians = np.arctan(vertical_distance / horizontal_distance)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees


    def detectPose(self, image_pose, pose, draw=False, display=False):
        original_image = image_pose.copy()
        
        image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
        
        resultant = pose.process(image_in_RGB)
        if resultant is None or resultant.pose_landmarks is None:
            return original_image, None 
        l = resultant.pose_landmarks.landmark
        if resultant.pose_landmarks and draw:    

            self.mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                    connections=self.mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255),
                                                                                thickness=3, circle_radius=3),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(49,125,237),
                                                                                thickness=2, circle_radius=2))

        if display:
            plt.figure(figsize=[22,22])
            plt.subplot(121)
            plt.imshow(image_pose[:,:,::-1])
            plt.title("Input Image")
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(original_image[:,:,::-1])
            plt.title("Pose detected Image");plt.axis('off')
        else:      
            return original_image, l

    def is_point_inside_rectangle(self, x1, y1, x3, y3, px, py):
        if (x1 <= px <= x3) and (y1 <= py <= y3):
            return True
        else:
            return False

    def get_left_and_right_angles(self, l):
        if l[15].x <= l[11].x:# or calculate_dist(l[11], l[13]) > 1.5* calculate_dist(l[13], l[15]):
            left = 0
        else:
            left = self.calculate_angle(l[11], l[13], l[15], thresh = 0.7)
        if l[12].x <= l[16].x:# or calculate_dist(l[12], l[14]) > 1.5* calculate_dist(l[14], l[16]):
            right = 0.0
        else:
            right = self.calculate_angle(l[12], l[14], l[16], thresh = 0.7)
        return left, right
    
    def get_left_and_right_angles_copy(self, l):
        if l[15].x <= l[11].x or l[15].y <= l[11].y:# or calculate_dist(l[11], l[13]) > 1.5* calculate_dist(l[13], l[15]):
            left = 180
        else:
            left = self.calculate_perpendicular_angle(l[13], l[15], thresh = 0.7, isleft = True)
        if l[12].x <= l[16].x or l[16].y <= l[12].y:# or calculate_dist(l[12], l[14]) > 1.5* calculate_dist(l[14], l[16]):
            right = 180
        else:
            right = self.calculate_perpendicular_angle(l[14], l[16], thresh = 0.7, isleft = True)
        return left, right

    def get_up_and_down_angles(self, l):
        if l[15].y > l[11].y:# or calculate_dist(l[11], l[13]) > 1.5* calculate_dist(l[13], l[15]):
            backward = 0.0
        else:
            backward = self.calculate_perpendicular_angle(l[13], l[15], thresh = 0.7)
        if l[16].y > l[12].y:# or calculate_dist(l[12], l[14]) > 1.5* calculate_dist(l[14], l[16]):
            forward = 0.0
        else:
            forward = self.calculate_perpendicular_angle(l[14], l[16], thresh = 0.7)
        return backward, forward

    def get_action(self, left, right, backward, forward):
        actions = [0]*4
        # if left >= 60 and left <= 120:
        #     actions[0] = 1
        # if right >= 60 and right <= 120:
        #     actions[1] = 1
        if left <= 50:
            actions[0] = 1
        if right <= 50:
            actions[1] = 1
        if backward >= 60:
            actions[2] = 1
        if forward >= 60:
            actions[3] = 1
        # print(actions)
        counts = sum(actions)
        if counts > 1:
            if counts == 2 and actions[2] == 1 and actions[3] == 1:
                return 4
            return -1
        elif counts == 1:
            return actions.index(1)
        return -2

    def get_stable_action(self, all_actions):
        element_count = {}
        for element in all_actions:
            if element in element_count:
                element_count[element] += 1
            else:
                element_count[element] = 1
        max_frequency_element = max(element_count, key=element_count.get)
        return max_frequency_element
        
    def do_action(self, action, last_entry, isdebug, time_gap = 1):
        if action == "left":
            if time.time() - last_entry[0] >= time_gap:
                last_entry[0] = time.time()
                # print("a")
                if not isdebug:
                    self.execute_command_in_robot("a")
        elif action == "right":
            if time.time() - last_entry[1] >= time_gap:
                last_entry[1] = time.time()
                # print("d")
                if not isdebug:
                    self.execute_command_in_robot("d")
        elif action == "backward":
            if time.time() - last_entry[2] >= time_gap:
                last_entry[2] = time.time()
                # print("s")
                if not isdebug:
                    self.execute_command_in_robot("s")
        elif action == "forward":
            if time.time() - last_entry[3] >= time_gap:
                last_entry[3] = time.time()
                # print("w")
                if not isdebug:
                    self.execute_command_in_robot("w")
        elif action == "stop":
            if time.time() - last_entry[4] >= time_gap:
                last_entry[4] = time.time()
                # print("v")
                if not isdebug:
                    self.execute_command_in_robot("v")
        return last_entry
    
    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, frame = self.video.read()
        if self.use_video_stream:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # face_width_in_frame, Faces, FC_X, FC_Y = self.face_data(frame, True, 0)
        # ret, frame = cap.read()

        if self.with_authentication:
            if self.prev_result and self.cnt % 5 != 0:
                result = self.prev_result
            else:
                result = DeepFace.verify(img1_path = self.owner, img2_path = frame, enforce_detection = False, model_name = "Facenet512")
        else:
            result = None

        l = None
        if self.with_authentication:
            if result is not None and result["distance"] < 0.45:
                img, l = self.detectPose(frame, self.pose_image, draw=True)
                rect = result["facial_areas"]["img2"]
                if rect is not None and l is not None:
                    if self.is_point_inside_rectangle(rect["x"],rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"], l[0].x * self.resolution[0], l[0].y * self.resolution[1]):
                        frame = img
                    frame = cv2.rectangle(frame, (rect["x"],rect["y"]), (rect["x"] + rect["w"], rect["y"] + rect["h"]), (0, 0, 255), 1) 
        else:
            frame, l = self.detectPose(frame, self.pose_image, draw=True)

        self.global_actions.append("None")
        if l is not None:
            left, right = self.get_left_and_right_angles_copy(l) #self.get_left_and_right_angles(l)
            backward, forward = self.get_up_and_down_angles(l)
            action = self.get_action(left, right, backward, forward)
            if action == -1:
                frame = cv2.putText(frame, f'action : {"choose only one direction"}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA, False)
            elif action >= 0:
                frame = cv2.putText(frame, f'action : {self.action_list[action]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA, False)
                self.global_actions[-1] = self.action_list[action]
        
        final_action = "None"
        if len( self.global_actions) >= self.buffer:
            final_action = self.get_stable_action(self.global_actions)
            self.global_actions =  self.global_actions[1:]

        self.last_entry = self.do_action(final_action, self.last_entry, self.isdebug)
        frame = cv2.putText(frame, f'stable action : {final_action}', (200, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv2.LINE_AA, False)
        # cv2.imshow('preview',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        self.cnt += 1
        self.prev_result = result

        # finding the distance by calling function Distance finder
        # for (face_x, face_y, face_w, face_h) in Faces:
        #     if face_width_in_frame != 0:
        #         # 1000 for laptop camera
        #         Distance = self.Distance_finder(
        #             700, 5.7, face_width_in_frame
        #         )
        #         Distance = round(Distance, 2)
        #         # Drwaing Text on the screen
        #         Distance_level = int(Distance)

        #         cv2.putText(
        #             frame,
        #             f"Distance {Distance} Inches",
        #             (face_x - 6, face_y - 6),
        #             cv2.FONT_HERSHEY_COMPLEX,
        #             0.5,
        #             ((0, 0, 0)),
        #             2,
        #         )

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(gen(VideoCameras()),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_command', methods=['POST'])
def send_command():
    command = request.form.get('command')
    print(command)
    
    execute_command_in_robot(command)
    
    return f"Command: {command} received in the backend"


from flask import Flask, request, jsonify


@app.route('/save_audio', methods=['POST'])
def save_audio():
    try:
        print('Saving the audio in the backend...')
        # Get the audio file from the request
        # audio_file = request.files['audio']
        audio_filepath = 'output.wav'
        print('Got here')
        # if audio_file:
            # Save the audio to a file named 'output.wav'
            # audio_file.save(audio_filepath)
        print('Running the whisper pipeline...')
        
        commands = whisper_pipeline(audio_filepath)
        print('COMMANDS: ', commands)
        
        commands = ['l', 'P', 'f'] + commands.split(' ')
        
        for command in commands:
            print(f'\nExecuting command {command}\n')
            execute_command_in_robot(command)
            time.sleep(4)
                
            return jsonify({"message": "Audio saved successfully. And commands run"}), 200
        else:
            return jsonify({"error": "No audio data received."}), 400
        

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


# import pyaudio
# import wave

# def record_audio(output_file, duration=10, sample_rate=44100, channels=1):
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=pyaudio.paInt16, channels=channels,
#                         rate=sample_rate, input=True,
#                         frames_per_buffer=1024)
#     frames = []

#     print("Recording audio...")
#     for i in range(0, int(sample_rate / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(data)

#     print("Finished recording.")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     wf = wave.open(output_file, 'wb')
#     wf.setnchannels(channels)
#     wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#     wf.setframerate(sample_rate)
#     wf.writeframes(b''.join(frames))
#     wf.close()
    



# # record_audio('output.wav', duration=10)


# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
# import openai

# def whisper_pipeline(audio_filepath):
#     openai.api_key = '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'

#     audio_file= open(audio_filepath, "rb")

#     transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']
#     # transcript = openai.Audio.transcribe("whisper-1", 'output.wav')

#     print('TRANSCRIPT: ', transcript)


#     prompt = f"""

#     Based on the mapping of characters to each of the keyboard movements: 
#     mapping = '2':'w', '4': 's', '1': 'a', '3':'d'.
#     You should build a sequence of movements by concatenating different 
#     characters based on the number of times each movement is mentioned 
#     in the <TRANSCRIPT> am gonna give you.

#     For instance: 1, 1, 3, 3, 3, 2, 2, 4
#     would be mapped into a result like: 'a a d d d w w s'

#     This is the <TRANSCRIPT>, which is the input to your response: {transcript} 

#     Just return the sequence of characters after mapping the transcript 
#     movements into characters. Do not return any programming code.

#     """


#     response = openai.Completion.create(
#                 engine="text-davinci-003",  # Use the appropriate engine
#                 prompt=prompt,
#                 max_tokens=50,  # You can adjust this to control response length
#             )

#     print('RESPONSE: ', response)

#     print(response['choices'][0])

#     return response['choices'][0]['text']

if __name__ == '__main__':
    app.run(debug=True, port = 5006)


