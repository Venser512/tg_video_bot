### TG_VIDEO_BOT 

## Inspired by https://github.com/Mozer/talk-llama-fast

## Requirements
- Windows 10/11 x64
- python, cuda
- 16 GB RAM
- Recommended: nvidia GPU with 12 GB vram.

## Installation
### For Windows 10/11 x64 with CUDA.
- Check that you have Cuda Toolkit 11.8. If not - install: https://developer.nvidia.com/cuda-11-8-0-download-archive
- python=3.11 

```
Clone this repository
git clone https://github.com/Venser512/tg_video_bot.git

cd tg_video_bot

python -m venv xtts
xtts\Scripts\activate

pip install git+https://github.com/Venser512/xtts-api-server pydub
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/Venser512/tts
pip install -r requirements.txt

xtts\Scripts\deactivate

Download content from  https://disk.yandex.ru/d/TwVGPR3ibYSQfg/tg_video_bot.zip and
extract in folder tg_video_bot

```

Update parameters in config.py
- bot_token = "Token of your telegram bot" (Ask Botfather, how to get it :)
- bot_user_id = Number (first part of bot_token) without quotation
- OPENROUTER_API_KEY = 'API KEY that you can get at openrouter.ai' (You can use it for free not so long time.)
- PROJECT_PATH = "Path when you installed this project\\\\tg_video_bot\\\\" (with double backslashes)
- conf_checkpoint_path='Path when you installed this project\\\\tg_video_bot\\\\checkpoints\\\\wav2lip.pth' (with double backslashes)


## Tests
- Run pytest in xtts enviroment

## Running
- Double click `xtts.bat` to start xtts server  NOTE: On the first run xtts will download DeepSpeed from github. If deepspeed fails to download "Warning: Retyring (Retry... ReadTimoutError...") - turn on VPN to download deepspeed (27MB) and xtts checkpoint (1.8GB), then you can turn it off). Xtts checkpoint can be downloaded without VPN. But if you interrupt download - checkpoint will be broken - you have to manually delete \xtts_models\ dir and restart xtts.
- Double click 'wav2lip.bat' to run FAST API server for wav2lip inference
- Double click 'tel.bat' to run telegram video bot

## Architecture and components
- telprod9.py (my own development) - Main python script that use threading. First thread interact with Telegram and put user questions in python queue. Second thread process the queue, asks LLM (by openrouter.ai API),
  then asks TTS (text-to-speach model) that convert LLM answer to audio, then asks lipsink model (Wav2lip) that convert audio to video answer and then send video answer to Telegram
- tel.bat - Command file that run telprod9.py in xtts enviroment
- TTS and XTTS_api_server - (opensource components) provide TTS (text-to-speach model) model answers by FAST API (forked from https://github.com/Mozer/xtts-api-server and https://github.com/Mozer/TTS)
- xtts_server.bat Command file that run XTTS_api_server in xtts enviroment 
- inference_new.py - (opensource component, part of Wav2lip model) (forked from https://github.com/Mozer/wav2lip with a steel file :) provide video answers (Lipsink)
- wav2lip_api.py - (my own development) FAST API for inference_new.py
- wav2lip.bat Command file that run wav2lip_api in xtts enviroment

## Design patterns
- Strategy (All persons that can interact with user are objects of class videobot). So when user choose person that he/she want to interact, it changes behaviour (LLM prompt, voice and video pattern)
- Decorators (Built in decorators in Fastapi and Telebot and my own decorator for running function with maximum 5 retries (if we have unsuccessful runs)
- Pydantic data model and validation (used in wav2lip_api.py for input data validation)

