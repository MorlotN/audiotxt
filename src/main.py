from datetime import timedelta
from fastapi import FastAPI, Request, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import stable_whisper
import srt
import ffmpeg
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

local_model_path = '/home/morlot/code/audiotxt/your_model_dir'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

whisper_model = stable_whisper.load_model("tiny")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    audio_file_path = 'audio.mp3'
    with open(audio_file_path, 'wb') as audio_file:
        audio_file.write(audio_data)

    # Transcription
    result = whisper_model.transcribe(audio_file_path, regroup=False)
    transcribed_text = result.text

    # Summarization
    inputs = tokenizer(transcribed_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the summary to a .txt file
    subtitle_file = "summary.txt"
    with open(subtitle_file, "w") as f:
        f.write(summary)

    # Provide the summary for download
    return StreamingResponse(open(subtitle_file, 'rb'), media_type="text/plain", headers={'Content-Disposition': 'attachment; filename="summary.txt"'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
