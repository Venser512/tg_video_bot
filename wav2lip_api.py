# -*- coding: cp1251 -*-

import inference_new
import json
from typing import Any, Dict, AnyStr, List, Union
from collections import OrderedDict
from fastapi import Depends, FastAPI, HTTPException, status, Request, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI()

class VideoRequest(BaseModel):
    input_video: str = Field(..., description="Path to the input video file")
    output_wav: str = Field(..., description="Path to the output audio file")
    output_video: str = Field(..., description="Path to the output video file")

@app.post("/video/")
async def root2(request: Request, video_request: VideoRequest):
    result = inference_new.wav2lip_main(
        face=video_request.input_video,
        audio_param=video_request.output_wav,
        outfile=video_request.output_video
    )

    content = {"Result": result}
    headers = {"Content-Language": "en-US"}

    return JSONResponse(content=content, headers=headers)




