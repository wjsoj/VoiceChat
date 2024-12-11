import os
from io import BytesIO
import base64
import httpx
import torch
from TTS.api import TTS
import whisper
from tempfile import NamedTemporaryFile

from openai import AsyncOpenAI

from chainlit.element import ElementBased
from chainlit.input_widget import Switch, Select
import chainlit as cl

# 添加说话人列表
SPEAKERS = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 
        'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 
        'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 
        'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 
        'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 
        'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 
        'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 
        'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 
        'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 
        'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 
        'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 
        'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']

cl.instrument_openai()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_BASE_URL"))

settings = None
tts_model = None
clone_file = None
current_language = "zh"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to encode an image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@cl.step(type="tool")
async def speech_to_text(audio_file):
    response = await client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return response.text


@cl.step(type="tool")
async def speech_to_text_offline(audio_file):
    global current_language
    # 初始化whisper模型（首次运行会下载模型）
    model = whisper.load_model("tiny")
    
    # 保存音频到临时文件
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_file[1])  # audio_file[1] 是音频数据
        temp_file_path = temp_file.name
    
    try:
        # 使用whisper进行识别
        result = model.transcribe(temp_file_path)
        current_language = result["language"]
        return result["text"]
    finally:
        # 清理临时文件
        os.unlink(temp_file_path)


@cl.step(type="tool")
async def generate_text_answer(transcription, images = None):
    if images:
        # Only process the first 3 images
        images = images[:3]

        images_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.mime};base64,{encode_image(image.path)}"
                },
            }
            for image in images
        ]

        model = "gpt-4-turbo"
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": transcription}, *images_content],
            }
        ]
    else:
        model = "gpt-4o-mini"
        messages = [{"role": "user", "content": transcription}]

    response = await client.chat.completions.create(
        messages=messages, model=model, temperature=0.3
    )

    return response.choices[0].message.content


@cl.step(type="tool")
async def text_to_speech(text: str, mime_type: str):
    audio_file = BytesIO()
    audio_file.name = "speech.mp3"
    
    response = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    # 将音频数据写入 BytesIO 对象
    for chunk in response.iter_bytes():
        audio_file.write(chunk)
    
    audio_file.seek(0)
    return audio_file.name, audio_file.read()


@cl.step(type="tool")
async def text_to_speech_offline(text: str):
    global tts_model, current_language, clone_file
    
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # 使用XTTS生成语音
        if clone_file:
            tts_model.tts_to_file(
                text=text,
                file_path=temp_file.name,
                language=current_language,
                speaker_wav=clone_file.path
            )
        else:
            tts_model.tts_to_file(
                text=text,
                file_path=temp_file.name,
                language=current_language,
                speaker=settings["speaker"],  # 使用选中的说话人
            )
        
        with open(temp_file.name, "rb") as audio_file:
            audio_data = audio_file.read()
        
        os.unlink(temp_file.name)  # 删除临时文件
        return "speech.mp3", audio_data

@cl.on_chat_start
async def start():
    # 初始化TTS模型
    global settings, tts_model
    
    settings = await cl.ChatSettings(
        [
            Switch(id="use_offline_tts", label="使用离线TTS", initial=True),
            Switch(id="use_offline_stt", label="使用离线语音识别", initial=False),
            Select(
                id="speaker",
                label="选择音色",
                values=SPEAKERS,
                initial_value="Ana Florence"
            ),
        ]
    ).send()
    
    if tts_model is None and settings["use_offline_tts"]:
        tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    res = await cl.AskActionMessage(
        content="Pick the TTS method:",
        actions=[
            cl.Action(name="Clone voice", value="clone", label="Clone"),
            cl.Action(name="Use default speakers", value="speaker", label="Speaker"),
        ],
    ).send()
    
    if res and res.get("value") == "clone":
        global clone_file
        files = None
        while files == None:
            files = await cl.AskFileMessage(
                content="Please upload a wav file to clone", accept={
                    "audio/wav": [".wav"]
                }
            ).send()

        clone_file = files[0]
        
        elements = [
            cl.Audio(path=clone_file.path, display="inline"),
        ]
        await cl.Message(
            content="Here is an audio file",
            elements=elements,
        ).send()
    else:
        await cl.Message(
            content="Welcome to the Chainlit audio example."
        ).send()


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # For now, write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_settings_update
async def on_settings_update(settings):
    global tts_model
    if tts_model is None and settings["use_offline_tts"]:
        tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

@cl.on_message
async def on_message(message: cl.Message):
    text_answer = await generate_text_answer(message.content)
    
    await cl.Message(
        content=text_answer
    ).send()
    
    use_offline_tts = settings["use_offline_tts"]
    
    if use_offline_tts:
        output_name, output_audio = await text_to_speech_offline(text_answer)
    else:
        output_name, output_audio = await text_to_speech(text_answer, "audio/mpeg")
    
    output_audio_el = cl.Audio(
        name=output_name,
        auto_play=True,
        mime="audio/mpeg",  # 固定使用 MP3 mime type
        content=output_audio,
    )
    answer_message = await cl.Message(content="").send()

    answer_message.elements = [output_audio_el]
    await answer_message.update()

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You", 
        type="user_message",
        content="",
        elements=[input_audio_el, *elements]
    ).send()
    
    
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    
    use_offline_stt = settings["use_offline_stt"]
    
    if use_offline_stt:
        transcription = await speech_to_text_offline(whisper_input)
    else:
        transcription = await speech_to_text(whisper_input)
    
    await cl.Message(
        author="You",
        type="user_message",
        content=transcription
    ).send()

    images = [file for file in elements if "image" in file.mime]

    text_answer = await generate_text_answer(transcription, images)
    
    await cl.Message(
        content=text_answer
    ).send()
    
    use_offline_tts = settings["use_offline_tts"]
    
    if use_offline_tts:
        output_name, output_audio = await text_to_speech_offline(text_answer)
    else:
        output_name, output_audio = await text_to_speech(text_answer, "audio/mpeg")
    
    output_audio_el = cl.Audio(
        name=output_name,
        auto_play=True,
        mime="audio/mpeg",  # 固定使用 MP3 mime type
        content=output_audio,
    )
    answer_message = await cl.Message(content="").send()

    answer_message.elements = [output_audio_el]
    await answer_message.update()