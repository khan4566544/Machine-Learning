import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import pyttsx3          # offline voice
import speech_recognition as sr
import datetime
import pywhatkit
import wikipedia
import pyjokes
import webbrowser
import subprocess
from win32com.client import Dispatch
import threading


def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()


def face_login():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    with open('data/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    print("🔒 Face Authentication Starting...")
    speak("Please look at the camera for face recognition")

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized = cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
            name = knn.predict(resized)[0]
            
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (50,255,50), 2)
            
            if knn.predict_proba(resized).max() > 0.7:
                cv2.putText(frame, "Authenticated!", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)
                cv2.imshow("AI Assistant - Face Login", frame)
                cv2.waitKey(1000)
                video.release()
                cv2.destroyAllWindows()
                speak(f"Welcome back {name}! Assistant activated.")
                return name

        cv2.imshow("AI Assistant - Face Login (Look at camera)", frame)
        if cv2.waitKey(1) == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            exit()


def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.pause_threshold = 1
        r.energy_threshold = 400
        audio = r.listen(source, timeout=5, phrase_time_limit=10)

    try:
        print("🔍 Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"You: {query}")
        return query.lower()
    except:
        return ""


def run_assistant(user_name):
    speak("Hello! I am your personal AI Assistant. How can I help you today?")

    while True:
        query = take_command()

        if "hello" in query or "hi" in query:
            speak(f"Hi ! How are you today?")

        elif "time" in query:
            time = datetime.datetime.now().strftime("%I:%M %p")
            speak(f"Current time is {time}")

        elif "date" in query:
            date = datetime.datetime.now().strftime("%B %d, %Y")
            speak(f"Today is {date}")

        elif "play" in query:
            song = query.replace("play", "")
            speak(f"Playing {song}")
            pywhatkit.playonyt(song)

        elif "search" in query or "who is" in query or "what is" in query:
            topic = query.replace("search", "").replace("who is", "").replace("what is", "")
            speak(f"Searching about {topic}")
            info = wikipedia.summary(topic, 2)
            print(info)
            speak(info)

        elif "joke" in query:
            joke = pyjokes.get_joke()
            print(joke)
            speak(joke)

        elif "open youtube" in query:
            webbrowser.open("youtube.com")
            speak("Opening YouTube")

        elif "open google" in query:
            webbrowser.open("google.com")
            speak("Opening Google")

        elif "open instagram" in query:
            webbrowser.open("instagram.com")
            speak("Opening Instagram")

        elif "shutdown" in query or "sleep" in query:
            speak("Goodbye! See you soon.")
            break

        elif "thank you" in query or "thanks" in query:
            speak("You're welcome! Always happy to help.")


if __name__ == "__main__":
    if not os.path.exists("data/names.pkl"):
        print("No face data found! Run the face collection script first.")
    else:
        authenticated_user = face_login()
        run_assistant(authenticated_user)