import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List, TypedDict
from segmentacion_model import predict_masks
import json

app = FastAPI()

class PromptData(TypedDict):
    box: tuple[int, int, int, int]
    frame: int

@app.post("/mask")
def segment_sequence(files: List[UploadFile], body: str = Form(...)):
    body_data = json.loads(body)
    
    box_data_list = []
    
    for d in body_data["annotations"]:
        box_data_list.append({
            "box": (d["box"][0], d["box"][1], d["box"][2], d["box"][3]),
            "frame": d["frame"]
        })
    
    masks = predict_masks(files, box_data_list)
    return masks

uvicorn.run(app, port=8001)