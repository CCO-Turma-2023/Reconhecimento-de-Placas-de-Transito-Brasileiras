classes = ['(A-18)', '(A-2a)', '(A-2b)', '(A-32a)', '(A-32b)', '(Del)', '(I-4)', '(MA)', '(MP-3)', '(R-1)', '(R-15)', '(R-19)30', '(R-19)40', '(R-19)50', '(R-19)60', '(R-19)70', '(R-19)80', '(R-19)90', '(R-2)', '(R-24a)', '(R-4a)', '(R-5a)', '(R-6a)', '(R-6c)']

from ultralytics import YOLO
import os

def TrainModel ():
    # Caminho absoluto para o data.yaml
    data_yaml = os.path.abspath("data.yaml")

    # Carregar modelo YOLOv8n pré-treinado no COCO
    model = YOLO("yolov8n.pt")

    # Treinar
    results = model.train(
        data=data_yaml,   # dataset (data.yaml)
        epochs=50,        # número de épocas
        imgsz=640,        # tamanho das imagens
        batch=16,         # tamanho do batch
        device=0          # usa GPU (coloque 'cpu' se não tiver GPU)
    )

    # Avaliar
    metrics = model.val()

if __name__ == "__main__":
    TrainModel()
