from roboflow import Roboflow
from credentials import ROBOFLOW_APIKEY

rf = Roboflow(api_key=ROBOFLOW_APIKEY)
project = rf.workspace("datasegmentation").project("shoes_segmentation")
version = project.version(5)
dataset = version.download("yolov8")
                