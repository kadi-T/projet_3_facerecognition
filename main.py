import os
import numpy as np
import face_recognition
from PIL import Image
import io
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Allow all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ImageUpload(BaseModel):
    filename: str
    content: bytes

img_folder = "img_upload"
img_path = "images"
os.makedirs(img_folder, exist_ok=True)
os.makedirs(img_path, exist_ok=True)

@app.post('/img_upload/')
async def recognition(file: UploadFile = File(...)):
    # Transformer img_upload en tableau numpy
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    query_img = np.array(image)
    
    # Redimensionner l'image pour réduire la taille
    query_img = cv2.resize(query_img, (0, 0), None, 0.25, 0.25)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # trouver le visage sur l'image
    facesCurrent = face_recognition.face_locations(query_img)
    # get Signatures from faces
    # on lui donne l'image redimensionnée et la localisation(coordonnées) du visage
    encodesCurrent = face_recognition.face_encodings(query_img, facesCurrent)
    
    signatures_class = np.load('FaceSignaturesF_db.npy')
    X = signatures_class[:, :-1].astype('float')
    Y = signatures_class[:, -1]
    
    results = []
    for encodeFace, faceloc in zip(encodesCurrent, facesCurrent):
        # comparaison de l'image new avec les images qu'on a dans notre db
        matches = face_recognition.compare_faces(X, encodeFace)

        # calcule de la distance de l'image new avec celles de notre db
        faceDis = face_recognition.face_distance(X, encodeFace)
        matchIndex = np.where(matches)[0]
        
        for index in matchIndex:
            image_name = Y[index].upper()
            # le chemin de l'image
            relative_path = os.path.join(img_path, image_name + ".jpg")
            # Vérifier si le fichier existe 
            if os.path.exists(relative_path):
                results.append({"image_name": image_name, "path": relative_path})
    
    if not results:
        return {"message": "No matches found"}
    
    return {"results": results}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

