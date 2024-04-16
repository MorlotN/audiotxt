from datetime import timedelta
from typing import Optional

from fastapi import FastAPI, Request, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ffmpeg
import numpy as np
import srt
import stable_whisper
# import openai
from openai import OpenAI
# client = OpenAI()

from langchain.chains.summarize import load_summarize_chain
# from langchain.llm import LocalLLM  # Supposé LLM local pour l'exemple
from langchain.chains import StuffDocumentsChain
# from langchain.llm.llm_chain import LLMChain
from langchain.prompts import PromptTemplate



app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
template = Jinja2Templates(directory='templates')

# OpenAI.api_key = "sk-WDi1GFIcaMY"

# Charger le modèle Whisper
model = stable_whisper.load_model("tiny")

def get_audio_buffer(filename: str, start: int, length: int):
    out, _ = (
        ffmpeg.input(filename, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, ss=start, t=length)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def transcribe_time_stamps(segments: list):
    return "\n".join([f"{seg.start} -> {seg.end}: {seg.text.strip()}" for seg in segments])

def make_srt_subtitles(segments: list):
    subtitles = [srt.Subtitle(index=i, start=timedelta(seconds=seg.start), end=timedelta(seconds=seg.end), content=seg.text.strip())
                 for i, seg in enumerate(segments, start=1)]
    return srt.compose(subtitles)

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return template.TemplateResponse('index.html', {"request": request})

@app.post('/download/')
async def download_subtitle(request: Request, file: bytes = File(), model_type: str = "tiny", timestamps: Optional[str] = Form("False"),
                            filename: str = "subtitles", file_type: str = "srt"):
    with open('audio.mp3', 'wb') as f:
        f.write(file)

    result = model.transcribe("audio.mp3", regroup=False)
    subtitle_file = f"{filename}.{file_type}"
    with open(subtitle_file, "w") as f:
        if file_type == "srt":
            f.write(make_srt_subtitles(result.segments) if timestamps == "True" else result.text)
        elif file_type == "vtt":
            f.write(result.to_vtt() if timestamps == "True" else result.text)
        elif file_type == "txt":
            f.write(transcribe_time_stamps(result.segments) if timestamps == "True" else result.text)

    return StreamingResponse(open(subtitle_file, 'rb'), media_type="application/octet-stream",
                             headers={'Content-Disposition': f'attachment;filename={subtitle_file}'})

# def summarize_text(text: str) -> str:
#     # Example using a hypothetical local LLM to summarize text
#     llm = LocalLLM(api_key="your_api_key_here")
#     prompt_template = "Summarize the following text: \"{text}\""
#     prompt = PromptTemplate.from_template(prompt_template)
#     llm_chain = LLMChain(llm=llm, prompt=prompt)
#     summary = llm_chain.run(text)
#     return summary


# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ]
# )

from openai import OpenAI
client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-4",
#   messages=[
#     {
#       "role": "system",
#       "content": "You will be provided with meeting notes, and your task is to summarize the meeting as follows:\n    \n    -Overall summary of discussion\n    -Action items (what needs to be done and who is doing it)\n    -If applicable, a list of topics that need to be discussed more fully in the next meeting."
#     },
#     {
#       "role": "user",
#       "content": "Fais moi compte rendu de ce text: {}\n\n".format(text)
#     }
#   ],
#   temperature=0.7,
#   max_tokens=64,
#   top_p=1
# )
def summarize_text(text: str) -> str:
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
          "content": "You will be provided with meeting notes, and your task is to summarize the meeting as follows:\n    \n    -Overall summary of discussion\n    -Action items (what needs to be done and who is doing it)\n    -If applicable, a list of topics that need to be discussed more fully in the next meeting."
        },
        {
          "role": "user",
          "content": "Fais moi compte rendu de ce text: {}\n\n".format(text)
        }
      ],
      temperature=0.7,
      max_tokens=64,
      top_p=1
    )
    # response = client.chat.completions.create(
    #   model="gpt-3.5-turbo",
    #     messages="Fais moi compte rendu de ce text: {}\n\n".format(text),
    #     max_tokens=150
    # )
    return response.choices[0].text.strip()

@app.post('/summarize/')
async def get_summary(request: Request, file: bytes = File(), model_type: str = "tiny"):
    with open('audio.mp3', 'wb') as f:
        f.write(file)

    transcription_result = model.transcribe("audio.mp3")
    summary = summarize_text(transcription_result.text)
    return {"transcription": transcription_result.text, "summary": summary}

@app.post('/transcribe_and_summarize/')
async def transcribe_and_summarize(file: bytes = File(), model_type: str = "tiny", summarize: bool = Form(False)):
    with open('temp_audio.mp3', 'wb') as f:
        f.write(file)

    transcription_result = model.transcribe("temp_audio.mp3")
    if summarize:
        summary = summarize_text(transcription_result.text)
        return {"transcription": transcription_result.text, "summary": summary}
    return {"transcription": transcription_result.text}

@app.post('/summarize_text/')
async def summarize_text_api(request: Request):
    data = await request.json()
    text = data['text']
    summary = summarize_text(text)
    return {"summary": summary}

