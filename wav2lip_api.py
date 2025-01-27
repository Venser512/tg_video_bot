# -*- coding: cp1251 -*-

import inference_new 

import json

from typing import Any, Dict, AnyStr, List, Union
from collections import OrderedDict

from fastapi import  Depends, FastAPI, HTTPException, status, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse


app = FastAPI()


@app.post("/video/")
async def root2 (request: Request):

    contents = await request.body()

    data = json.loads(contents)

    result = inference_new.wav2lip_main(face = data['input_video'], audio_param = data['output_wav'], outfile = data['output_video'])

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}

    return JSONResponse(content=content, headers=headers)





