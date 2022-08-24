import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from utils.logodetect import logodetection

app = FastAPI()
logomodel = logodetection()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """
    Just the root path
    """
    return {"Hello": "World"}

@app.post("/LOGOS")
async def get_output(score: float=Form(), file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = logomodel.predict(image=img, score=score)

    image = img
    for box in boxes:
        image = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, im_png = cv2.imencode(".png", image)

    png_as_text = base64.b64encode(im_png)

    return {
        "bounding_boxes" : boxes,
        "image" : png_as_text,
        "score" : score
    }

if __name__=="__main__":
    uvicorn.run(app, port=8888, host='127.0.0.1')