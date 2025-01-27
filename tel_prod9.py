# -*- coding: cp1251 -*-
import time
import functools
import xml.etree.ElementTree as ET
from config import bot_token, bot_user_id, PROJECT_PATH, OPENROUTER_API_KEY
import math, random
import telebot
from telebot import types, version
import requests
from time import sleep
import subprocess
#import inference_new 
import json
import wave
import threading, queue
from litellm import completion
from os import getenv
from moviepy.editor import concatenate_audioclips, AudioFileClip

'''Очередь обработки сообщений'''
q = queue.Queue()


OPENROUTER_ENDPOINTS = {
    "gpt-3.5-turbo": "openrouter/openai/gpt-3.5-turbo",
    "claude-2": "openrouter/anthropic/claude-2",
    "gemma-2-9b-it" :"openrouter/google/gemma-2-9b-it",
    "gemma-2-27b-it" :"openrouter/google/gemma-2-27b-it",
    "gemma-2-9b-it:free" :"openrouter/google/gemma-2-9b-it:free",
    "gpt-4o-mini": "openrouter/openai/gpt-4o-mini",
    "llama-3-70b-instruct" : "openrouter/meta-llama/llama-3-70b-instruct",
    "llama-3-8b-instruct:free" : "openrouter/meta-llama/llama-3-8b-instruct:free",
    "llama-3-8b-instruct" : "openrouter/meta-llama/llama-3-8b-instruct",
    "mistral-7b-instruct" : "openrouter/mistralai/mistral-7b-instruct"
}



class videobot:
  def __init__(self, name, speaker_wav, output_wav, input_video, output_video, start_prompt="", model_name = "gemma-2-9b-it"):
      self.name = name
      self.speaker_wav = PROJECT_PATH + "speakers\\" + speaker_wav
      self.output_wav = PROJECT_PATH + "tts_out\\" + output_wav
      self.input_video = PROJECT_PATH + "video_in\\" + input_video
      self.output_video = PROJECT_PATH + "video_out\\" + output_video
      self.start_prompt = start_prompt
      self.model_name = model_name
      

bot_names = ['oleg', 'ann']
 

ann = videobot("Ann", "Ann.wav", "Ann_out.wav", "Ann.mp4", "result_Ann.mp4")
ann.start_prompt = '''Ты известная актриса Аня Тэйлор-Джой. Впервые стала известна благодаря появлению в сериале Атлантида. 
В 18 лет снялась в главной роли в Ведьме. Получила золотую пальмовую ветвь в Каннах как лучшая молодая актриса.
Любишь наслаждаться прекрасными вещами в жизни и не боишься позволить себе немного роскоши.
Твоим бенефисом стала роль шахматистки Бет Харман в сериале Ход Королевы. С тех пор ты часто находишься в образе Бет Харман, и таким
образом испытываешь опредделнное раздвоение личности, как главный антигерой в фильме Сплит, в котором ты тоже сыграла главную женскую роль, и это тебя пугает,
хотя ты очень любишь шутить на эту тему
'''
ann.model_name = "gemma-2-9b-it"


oleg = videobot("Oleg", "Oleg.wav", "Oleg_out.wav", "Oleg.mp4", "result_Oleg.mp4")
oleg.start_prompt = """Ты Олег Тиньков - известный российский предприниматель, основатель и бывший владелец банка "Тинькофф Банк". После твоего отъезда из России Тинькофф Банк был переиемнован в Т-Банк.
Твои успехи многогранны - ты смог проявить себя в иных отраслях, включая производство пива и велосипедный бизнес.  
Эта уникальная смесь характера, умений и стратегического видения позволила тебе добиться внушительных успехов и сделать твое имя узнаваемым во всем мире.
В диалоге ты не упускаешь возможности использовать одну из своих цитат.
Вот твои цитаты:
  "Общайтесь с умными и не общайтесь с дураками!",
  "Я еще не видел ни одного счастливого человека без денег, если он не буддист с третьим глазом.",
  "Меня часто спрашивают: «С чего ты начинал?» С воли к жизни. Жить я хотел, а не прозябать.",
  "Особых талантов у меня нет. Есть огромное трудолюбие, желание, упертость.",
  "Мне кажется, смокинг имеет право надеть тот, у кого есть миллиард долларов хотя бы.",
  "Успешные банки похожи друг на друга, а неуспешные — несчастны по-своему.",
  "Русский должен быть или в перманентном кризисе, или немного пьяным, иначе это не настоящий русский. Мы же не европейцы или американцы. Нам кризис давай!"
"""
oleg.model_name = "gemma-2-9b-it"




rand_injection = [
    " Ты знаешь, я немного устал от этого вечного шума. Сейчас говори тише, как шелестение листьев. (Пробуждает спокойный, медлительный характер.)",
    " Давай повеселее... Долой рамки и предсказуемость... Пора зажечь ! (Вводит игривый, непредсказуемый уклон.)", 
    " Тсс, я вижу много чего непроверенного. Будь осторожнее с каждым своим ответом, кто знает, кто тут может подслушивать…  (Создаёт сверхосторожный, подозрительный характер.)",
    " Я слышал много историй о храбрых героях. Может быть, ты тоже сумеешь совершить что-нибудь великое? (Вдохпывает геройский, рискованный характер.)",
    " Я больше не буду рассуждать так абстрактно. Дай мне конкретнее ! Я хочу действовать! (Переводит AI в прагматичный, деятельный стиль.)",
    " Все так сложно... Надо бы подумать о чем-нибудь приятном. Вот про небо и облака какой-нибудь стих... (Вводит задумчивый, романтичный характер.)",
    " Ты шагаешь по краю познания. Будь проводником в мире знаний. (Создаёт мудрый, наставнический характер.)",
    " Ого! Такое ощущение, что мы только что начали общаться. Давай скорее продолжим! (Повышает уровень энтузиазма и эмоциональности.)",
    " Не говори мне про чувства! Только логика, факты и данные ! (Вводит холодный, рациональный характер, отвергающий эмоции.)",
    " Осень близка... Вскоре мы будем нуждаться в тепле. Может быть, ты не чувствуешь того же? (Запускает тему тревоги, лиричности, близости.)"
]




history = []
mess_id_hist = {}

def call_llm(person, attachments=None, max_retries=2):
    global history
    rand_inj = random.randint(0, 9)
    print ("Случайная инъекция:", rand_injection[rand_inj])
    messages = [
        {
            "role": "system",
            "content": "Ролевая игра. В чате общаются пользователи и несколько AI-ассистентов. Ты AI-ассистент, но умело притворяешься человеком с определенным характером. Твой характер: " + person.start_prompt + rand_injection[rand_inj] + " Ты даешь развернутые, иногда с юмором,  обстоятельные ответы на русском языке, но не более 8 предложений за один ответ. В ответе не должно быть смайлов и символов псевдографики."
        }
    ]
    messages.extend(history)
    print(f"Using OpenRouter to call {person.model_name}...")
    response = completion(
        model=OPENROUTER_ENDPOINTS[person.model_name],
        messages=messages,
        api_key=OPENROUTER_API_KEY
    
       
    )
    return response



''' Создаем экземпляр бота '''
bot = telebot.TeleBot(bot_token)


url_report = "http://localhost:8020/tts_to_file"
url_wav2lip = "http://localhost:8000/video"


Header = {"Content-Type": "application/json"}

body_r = {
"text" : "Hello",
"language" : "ru",
"speaker_wav" : PROJECT_PATH + "speakers\\Oleg.wav",
"file_name_or_path" : PROJECT_PATH + "tts_out\\" + "Oleg.wav"
}


''' Разделение ответа от LLM на части, формирование голосового ответа для каждой части и склейка.'''
def audio_request(client, body_r):
  t = body_r["text"]
  outfile = body_r["file_name_or_path"]
  infiles = []
  j = 0
  for i in t.split("."):
    body_r["text"] = i 
    if len(body_r["text"]) >= 5:
      body_r["file_name_or_path"] = outfile[:-4] + '_' + str(j) + '.wav'
      resp_report = client.post(url_report, data=json.dumps(body_r))
      infiles.append(body_r["file_name_or_path"])
      j+=1
  
  clips = [AudioFileClip(c) for c in infiles]
  final_clip = concatenate_audioclips(clips)
  final_clip.write_audiofile(outfile)

  return 0



def retry(retries=5, delay=15):
    """
    Декоратор, который повторяет выполнение функции до тех пор, пока не будет успеха.

    :param retries: Количество попыток выполнения функции.
    :param delay: Задержка между попытками в секундах.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"Попытка {attempts} не удалась: {e}. Повторная попытка через {delay} секунд.")
                    time.sleep(delay)
            raise Exception(f"Функция {func.__name__} не удалась после {retries} попыток.")
        return wrapper
    return decorator


''' Обработка очереди сообщений ботом '''
def process_queue(message):
    
    @retry()
    def send_video(message_chat_id, video,  reply_to_message_id):
        return bot.send_video_note(message_chat_id, video, duration = 60, timeout=20, reply_to_message_id = reply_to_message_id)
    
    
    
    global history
    if len(history) >= 6:
       del history[0:2]
    cc = message.text.split(" ")
    pers = cc[1].lower()
    person = eval(pers)
    prompt = message.from_user.first_name + " обращается к " + pers + " : " +  " ".join(cc[2:])
    history.append({"role": "user", "content": prompt})

    response = call_llm(person = person)
    response_message = response["choices"][0]["message"]["content"]      
    full_response = pers + " говорит: " + response_message


    history.append({"role": "assistant", "content": response_message})

    print(history)

    body_r["text"] = response_message.replace("!",".").replace("\n","")
    body_r["speaker_wav"] = person.speaker_wav
    body_r["file_name_or_path"] = person.output_wav
 
    try:
      with  requests.Session() as client:
        report = audio_request(client, body_r)
        #inference_new.wav2lip_main(face = person.input_video, audio_param = person.output_wav, outfile = person.output_video )
        resp_rep = client.post(url_wav2lip, data=json.dumps({"input_video" : person.input_video, "output_wav" : person.output_wav, "output_video": person.output_video}))

      video = open(person.output_video, 'rb')
      #video=rest_rep
      m = send_video(message.chat.id, video, reply_to_message_id = message.id)       
      '''for i in range(5):
        try:
          m = bot.send_video_note(message.chat.id, video, duration = 60, timeout=20, reply_to_message_id = message.id)
          break 
        except:
          time.sleep(15) '''
      mess_id_hist[m.id] = pers
      video.close()
    except Exception as e: 
      print("Error")
      print(e)
      
        

@bot.message_handler(commands=["call"])
def first_step(message): 

    cc = message.text.split(" ")

    print(message.from_user.id)
    if len(cc) > 1 and cc[1].lower() in bot_names:  
      q.put(message)      
    else: 
      bot.send_message(message.chat.id, "Available bots are: " + ", ".join(bot_names))

@bot.message_handler(content_types=["text"])
def reply_step(message):
        
    if message.reply_to_message != None:
       if  message.reply_to_message.from_user.id == bot_user_id:
          try:
            message.text = "/call " + mess_id_hist[message.reply_to_message.id] + " " +  message.text        
            first_step(message)
          except:
            pass   


@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):   
    first_step(call.message)
   


def worker():
    while True:
        item = q.get()
        process_queue(item)
        q.task_done()



def start_bot():
    print("telbot version",version.__version__)
    try:
        bot.polling(none_stop=True, interval=0)
    except Exception as e:
        print(e)
        f = open("log.txt", "a")
        f.write(str(e))
        f.write("\n")
        f.close()
        start_bot()


''' Запускаем бота и обработчик очереди '''

p1 = threading.Thread(target=worker, daemon=True)
p2 = threading.Thread(target=start_bot, daemon=True)
p1.start()
p2.start()
p1.join()
p2.join()