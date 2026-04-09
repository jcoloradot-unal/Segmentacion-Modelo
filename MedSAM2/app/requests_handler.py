import uvicorn
from fastapi import FastAPI, UploadFile, Form
from typing import List, TypedDict
from segmentacion_model import predict_masks_with_points
import json

app = FastAPI()


class PromptData(TypedDict):
    box: tuple[int, int, int, int]
    frame: int


class PointPromptData(TypedDict):
    points: List[tuple[int, int]]
    labels: List[int]
    frame: int


@app.get("/")
def test():
    return "that was succesful :D"


@app.post("/mask/points")
def segment_sequence_with_points(files: List[UploadFile], body: str = Form(...)):
    body_data = json.loads(body)

    point_data_list = []

    for d in body_data["annotations"]:
        # Convert points list of lists to list of tuples
        points = [tuple(point) for point in d["points"]]

        point_data_list.append(
            {"points": points, "labels": d["labels"], "frame": d["frame"]}
        )

    masks = predict_masks_with_points(files, point_data_list)
    js = {"masks": masks}

    print(str(js)[:200])

    return js


uvicorn.run(app, host="0.0.0.0", port=8001)
