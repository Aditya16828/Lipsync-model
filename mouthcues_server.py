from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import librosa
import numpy as np
from tensorflow import keras
from google.cloud import dialogflow
from pydantic import BaseModel
import base64

MOUTHCUES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X']
project_id = 'chatbot-iioj'

app = FastAPI()
session_client = dialogflow.SessionsClient()
# session = session_client.session_path(project_id, "*")
model = keras.models.load_model('./saved_models/8.keras')

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryModel(BaseModel):
    text: str
    userid: str | None = None

def detect_intent_texts(project_id, session_id, texts, language_code):
    session = session_client.session_path(project_id, session_id)
    # print(f"Session path: {session}\n")

    for text in texts:
        text_input = dialogflow.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.QueryInput(text=text_input)
        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )
        # print(response)
        return response.output_audio

def getMouthCues(audioData, sampleRate):
    mouthCues = []
    print(f"Duration: {len(audioData)/sampleRate} secs")
    duration = len(audioData)/sampleRate

    delta = 0.1

    start = 0 # sec
    end = delta # sec

    while(end <= duration):
        audioChunk = audioData[int(start*sampleRate):int(end*sampleRate)-1]
        mfccs = np.mean(librosa.feature.mfcc(y=audioChunk, sr=sampleRate, n_mfcc=20, n_fft=256, hop_length=64).T, axis=0)
        prediction = MOUTHCUES[model.predict(mfccs.reshape(1, -1)).argmax(axis=1)[0]]
        mouthCues.append({"start": start, "end": end, "value": prediction})
        start = end
        end += delta
    
    finalRes = []
    start = mouthCues[0]['start']
    end = mouthCues[0]['end']
    for idx in range(1, len(mouthCues)):
        if(mouthCues[idx]['value'] != mouthCues[idx-1]['value']):
            # print(mouthCues[idx-1]['value'], start, end)
            finalRes.append({"value": mouthCues[idx-1]['value'], "start": start, "end": end})
            start = mouthCues[idx]['start']
            end = mouthCues[idx]['end']
        else:
            end = mouthCues[idx]['end']
    
    return finalRes

@app.post("/predict")
async def predict(
    text: QueryModel
):
    audio = detect_intent_texts(project_id, text.userid, [text.text], "en")
    audioData, sr = librosa.load(io.BytesIO(audio))
    mouthCues = getMouthCues(audioData, sr)
    return {"mouthCues": mouthCues, "audio": base64.b64encode(audio).decode('utf-8')}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)