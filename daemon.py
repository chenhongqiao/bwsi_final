from torchtext import vocab
from torch import nn
from face_emotion.screen_grabber import start_emotion_detection
import time
import multiprocessing
import speech_recognition as sr
import openai
from gtts import gTTS
from playsound import playsound
import PySimpleGUI as sg
from database import Database

openai.api_key = "CHANGEME"
openai.organization = "CHANGEME"


def create_window():
    window = sg.Window(
        title="Virtual Assitant",
        layout=[[sg.Image("microphone.png", expand_x=False, expand_y=True)]],
        margins=(100, 0),
        location=(0, 0),
    )
    while True:
        window.read()


def create_namespace():
    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    namespace.emotion = "Neutral"
    namespace.emotion_embedding = None
    namespace.window_title = "Virtual Assistant"

    detection = multiprocessing.Process(
        target=start_emotion_detection, args=(namespace,)
    )

    detection.start()
    return namespace


if __name__ == "__main__":
    db = Database()
    namespace = create_namespace()

    window = multiprocessing.Process(target=create_window)
    r = sr.Recognizer()
    with sr.Microphone(device_index=3) as source:
        print("ready")
        messages = [
            {
                "role": "system",
                "content": """You are Alexa, a virtual assitant designed to listen to music with the user together. You can activate a GUI and interact with an external recommendation system by outputting. The first line of your output must be a command, and your second line is the message send to user.
1. When the conversation first begins, append [GUI] exactly as written to your response to activate a GUI. Regardless of what the user asks, you must do this every time on your first response.
Example:
User: Hi Alexa!
Assitant: [GUI]
Hey! I'm here.
2. When asked by the user to recommend a song, you must output [SONG REC] exactly as written to activate an external song recommendation system. The recommendation system will tell you the name of the song to recommend in the next message. Then, you must use [PLAY] command to start playing. If the user asks you to stop playing, use the [STOP] command.
Example:
User: Can you play me a song?
Assitant: [SONG REC] 
Sure! Let me find a song from your personal library.
User: <SYSTEM>Faded</SYSTEM>
Assitant: [PLAY]
Now playing Faded. Hope you will enjoy.
User: Stop the music
Assitant: [STOP]
Let me stop the song.
3. In scenarios not described above, output [NO COMMAND] in the first line.
""",
            },
        ]

        skip = False
        current_song = ""
        player = None
        activated = False
        while True:
            if not skip:
                audio = r.listen(source, phrase_time_limit=4)
                try:
                    text = r.recognize_google(audio, language="en-US")
                except:
                    print("No content")
                    continue
                print(text)
                messages.append({"role": "user", "content": text})
            skip = False
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", temperature=1, n=1, messages=messages
            )
            print(response["choices"][0]["message"])
            messages.append(response["choices"][0]["message"])
            command, text = response["choices"][0]["message"]["content"].split(
                "\n", maxsplit=2
            )

            command = command.strip()
            if command == "[GUI]":
                activated = True
                window.start()
                time.sleep(3)

            print("ok")

            if activated:
                audio = gTTS(text=text, lang="en", slow=False)
                audio.save("assistant.mp3")
                playsound("assistant.mp3")
                if command == "[SONG REC]":
                    song = db.query(namespace.emotion_embedding)
                    messages.append({"role": "user", "content": song})
                    current_song = song
                    skip = True
                elif command == "[PLAY]":
                    player = multiprocessing.Process(
                        target=playsound, args=(f"songlib/{current_song}",)
                    )
                    player.start()
                elif command == "[STOP]":
                    player.terminate()
                print(text)
                print((command, text))
