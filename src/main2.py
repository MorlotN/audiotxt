from datetime import timedelta
from typing import Optional

from fastapi import FastAPI, Request, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ffmpeg
import numpy as np
import srt as srt
import stable_whisper


def get_audio_buffer(filename: str, start: int, length: int):
    """
    input: filename of the audio file, start time in seconds, length of the audio in seconds
    output: np array of the audio data which the model's transcribe function can take as input
    """
    out, _ = (
        ffmpeg.input(filename, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, ss=start, t=length)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe_time_stamps(segments: list):
    """
    input: a list of segments from the model's transcribe function
    output: a string of the timestamps and the text of each segment
    """
    string = ""
    for seg in segments:
        string += " ".join([str(seg.start), "->", str(seg.end), ": ", seg.text.strip(), "\n"])
    return string


def make_srt_subtitles(segments: list):
    subtitles = []
    for i, seg in enumerate(segments, start=1):
        start_time = seg.start
        end_time = seg.end
        text = seg.text.strip()

        subtitle = srt.Subtitle(
            index=i,
            start=timedelta(seconds=start_time),
            end=timedelta(seconds=end_time),
            content=text
        )
        subtitles.append(subtitle)

    return srt.compose(subtitles)


app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')
template = Jinja2Templates(directory='templates')


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return template.TemplateResponse('index.html', {"request": request, "text": None})


@app.post('/download/')
async def download_subtitle(
        request: Request,
        file: bytes = File(),
        model_type: str = "tiny",
        timestamps: Optional[str] = Form("False"),
        filename: str = "subtitles",
        file_type: str = "srt"
):
    # Save the uploaded file
    with open('audio.mp3', 'wb') as f:
        f.write(file)

    # Load the model and transcribe the audio
    model = stable_whisper.load_model(model_type)
    result = model.transcribe("audio.mp3", regroup=False)

    # Define the subtitle file based on selected format
    subtitle_file = f"{filename}.{file_type}"
    with open(subtitle_file, "w") as f:
        if file_type == "srt":
            if timestamps == "True":
                f.write(make_srt_subtitles(result.segments))
            else:
                f.write(result.text)
        elif file_type == "vtt":
            if timestamps == "True":
                f.write(result.to_vtt())
            else:
                f.write(result.text)
        elif file_type == "txt":
            if timestamps == "True":
                f.write(transcribe_time_stamps(result.segments))
            else:
                f.write(result.text)

    # Create a streaming response with the file
    media_type = "application/octet-stream"
    response = StreamingResponse(
        open(subtitle_file, 'rb'),
        media_type=media_type,
        headers={'Content-Disposition': f'attachment;filename={subtitle_file}'}
    )

    return response

def summarize_text(text: str) -> str:
    # Supposons que vous ayez un LLM local disponible ici
    # Utilisation fictive d'un modèle de résumé local
    # summary = local_llm.summarize(text)
    summary = text+"coucoucocucocu"
    return summary

@app.post('/summarize/')
async def get_summary(
        request: Request,
        file: bytes = File(),
        model_type: str = "tiny"
):
    # Enregistrer et transcrire comme précédemment
    with open('audio.mp3', 'wb') as f:
        f.write(file)

    model = stable_whisper.load_model(model_type)
    result = model.transcribe("audio.mp3")

    # Obtenir le résumé du texte transcrit
    summary = summarize_text(result.text)

    # Retourner le résumé dans la réponse
    return {"summary": summary}
