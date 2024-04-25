from fastapi import FastAPI, Request, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from transformers import AutoModelForCausalLM, AutoTokenizer
import stable_whisper
import ffmpeg
import numpy as np

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
template = Jinja2Templates(directory='templates')

local_model_path = '/home/morlot/code/audiotxt/your_model_dir'
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

whisper_model = stable_whisper.load_model("tiny")

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return template.TemplateResponse('index.html', {"request": request})

@app.post('/process_audio/')
async def process_audio(request: Request, file: bytes = File(...)):
    # Save the audio file
    audio_filename = 'temp_audio.mp3'
    with open(audio_filename, 'wb') as audio_file:
        audio_file.write(file)

    # Transcribe audio
    transcription_result = whisper_model.transcribe(audio_filename, regroup=True)
    transcription_text = "\n".join([seg['text'] for seg in transcription_result.segments])

    # Generate summary
    inputs = tokenizer(transcription_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save summary to text file
    summary_filename = 'summary.txt'
    with open(summary_filename, 'w') as summary_file:
        summary_file.write(summary_text)

    return StreamingResponse(open(summary_filename, 'rb'), media_type='text/plain', headers={'Content-Disposition': f'attachment; filename={summary_filename}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
