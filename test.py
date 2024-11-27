import cv2, os, argparse, random
import numpy as np
import torchvision.transforms as transforms

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

class ImglistOrdictToTensor(torch.nn.Module):
    """
    Converts a list or a dict of numpy images to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH).
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list_or_dict):
        """
        Converts each numpy image in a list or a dict to
        a torch Tensor and stacks them into a single tensor.

        Args:
            img_list_or_dict: list or dict of numpy images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        if isinstance(img_list_or_dict, list):
            return torch.stack([transforms.functional.to_tensor(img)
                                for img in img_list_or_dict])
        else:
            return torch.stack([transforms.functional.to_tensor(img_list_or_dict[k])
                                for k in img_list_or_dict.keys()])

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

# Parametri per la transform
side_size=256
crop_size=224
alpha=4
num_frames=32
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

num_frame=0 #frame nel quale viene catturato il fuoco
counter_consecutive_fire=0 #numero di frame consecutivi di fuoco identificati dalla rete
fire_detection=3 #numero di occorrenze consecutive della rete nell'individuazione del fuoco per ottenere un risultato positivo
img_counter=0 #immagini attualmente messe nella lista
res=0 #risultato della rete
img_num=32 #numero di immagini da mettere in una lista
intersect_frames=0 #numero di frames che si riutilizzano tra un controllo e l'altro
imgs=[] #lista immagini
img_min_limit=fire_detection-1 #numero minimo di risultati positivo se il video termina per dare risultato positivo
last_res=[-1]*img_min_limit #lista contentente gli ultimi risultati 

# Preparazione modello, caricando i pesi e mettendolo in modalità di valutazione
model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
model.blocks[-1].proj = torch.nn.Sequential(torch.nn.Linear(2304,128),torch.nn.ReLU(),torch.nn.Linear(128,2))
model.blocks[-1].output_pool=  torch.nn.Softmax()
model.load_state_dict(torch.load("model.pth"))
model.eval()
model.cuda()

def inference(frames):
    global last_res
    global counter_consecutive_fire
    global fire_detection
    # Preparazione input
    images =  ImglistOrdictToTensor.forward(frames)
    frames_tensor = {'video': images.permute(1,0,2,3)}
    video_data = transform(frames_tensor)
    inputs = video_data["video"]
    inputs = [i[None, ...].cuda() for i in inputs]
    # Generazione del risultato e relativa valutazione
    with torch.no_grad():
        output = model(inputs)
        output=output.squeeze(dim=-1)
        res = output[:, 1] >= 0.5
        res=int(res)
        last_res.pop(0)
        last_res.append(res)
        if res==1:
          counter_consecutive_fire+=1
        else:
            counter_consecutive_fire=0
        if counter_consecutive_fire==fire_detection:
            return 1
        else:
            return 0
        

################################################

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(os.path.join(args.videos, video))
    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        if ret: # Se tutto è andato bene
          num_frame+=1
          imgs.append(img)
          img_counter+=1
          # Quando sono stati raccolti i frames necessari, esegui la valutazione
          if img_counter==img_num:
            res=inference(imgs)
            img_counter-=(img_num-intersect_frames)
            imgs = imgs[img_num-intersect_frames:]
        # Gestione caso particolare frame finiti
        elif last_res.count(1)==img_min_limit:
          res=1
        # Gestione caso particolare pochi frame in video
        elif last_res.count(1)==(img_min_limit-1) and last_res.count(-1)==1:
          res=1
        # Se è stato trovato il fuoco
        if res==1:
            break
        ########################################################
    
    cap.release()
    f = open(args.results+video+".txt", "w")
    # Here you should add your code for writing the results
    pos_neg = res #qui scrivi se è 1 o 0 (fuoco o no)
    if pos_neg:
        t = num_frame #qua devi scrivere in che momento (metti numero frame)
        f.write(str(t))
    #reset dei vari parametri
    num_frame=0
    res=0
    counter_consecutive_fire=0
    img_counter=0
    imgs.clear()
    last_res.clear()
    last_res=[-1]*img_min_limit
    ########################################################
    f.close()