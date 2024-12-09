# VoiceChat-Chainlit

chainlit cookbook中的audio-assistant已被弃用，改用OpenAI新发布的Realtime API，但完成度并不好。本项目是基于先前audio-assistant的优化版实现，支持全程调用OpenAi whisper+tts或使用离线模型（`whisper`+`coqui-tts`）

## Usage

```bash
git clone git@github.com:wjsoj/VoiceChat.git
cd VoiceChat
python -m venv voicechat
source voicechat/bin/activate
pip install -r requirements.txt
chainlit run app.py -w
```

如果使用OpenAI服务，请参照`.env.example`添加环境变量

## requirements

chainlit新版本上游代码存在问题，需要锁pydantic版本避免报错

```
chainlit
openai
pydantic==2.10.1
openai-whisper
coqui-tts
pypinyin
```