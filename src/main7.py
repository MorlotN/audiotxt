from datetime import timedelta
from typing import Optional

import numpy as np
import ffmpeg
import srt
import uvicorn
from fastapi import FastAPI, Request, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from transformers import AutoModelForCausalLM, AutoTokenizer
import stable_whisper

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
template = Jinja2Templates(directory='templates')

# Chemin local pour le modèle et tokenizer
local_model_path = '/home/morlot/code/audiotxt/your_model_dir'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Charger Whisper model pour la transcription audio
whisper_model = stable_whisper.load_model("tiny")

def get_audio_buffer(filename: str, start: int, length: int):
    out, _ = ffmpeg.input(filename, threads=0).output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, ss=start, t=length).run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def make_srt_subtitles(segments: list):
    return srt.compose([srt.Subtitle(index=i, start=timedelta(seconds=seg.start), end=timedelta(seconds=seg.end), content=seg.text.strip()) for i, seg in enumerate(segments)])

def transcribe_time_stamps(segments: list):
    return "\n".join([f"{seg.start} -> {seg.end}: {seg.text.strip()}" for seg in segments])

def summarize_text(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return template.TemplateResponse('index.html', {"request": request})

@app.post('/download/')
async def download_subtitle(request: Request, file: bytes = File(), filename: str = "subtitles", file_type: str = "txt"):
    with open('audio.mp3', 'wb') as f:
        f.write(file)
    result = whisper_model.transcribe("audio.mp3", regroup=False)
    subtitle_file = f"{filename}.{file_type}"
    with open(subtitle_file, "w") as f:
        if file_type == "srt":
            f.write(make_srt_subtitles(result.segments))
        elif file_type == "vtt":
            f.write(result.to_vtt())
        elif file_type == "txt":
            f.write(transcribe_time_stamps(result.segments))
    return StreamingResponse(open(subtitle_file, 'rb'), media_type="application/octet-stream", headers={'Content-Disposition': f'attachment;filename={subtitle_file}'})

@app.post('/summarize_text/')
async def summarize_text_api(request: Request, file: bytes = File(), filename: str = "summary.txt"):
    data = await request.json()
    text = data.get('text', '')
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for summarization.")
    summary = summarize_text(text)
    return {"summary": summary}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
