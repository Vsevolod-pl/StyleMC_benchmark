import torch
from argparse import Namespace
from torchvision import transforms
from PIL import Image

from models.encoders import psp_encoders


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

model_path='./e4e_ffhq_encode.pt'
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)

encoder = psp_encoders.Encoder4Editing(50, 'ir_se', opts)
encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
encoder.cuda()

img_tr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def calc_latent(image_path, latent_avg, align=True, device='cuda'):
    encoder.to(device)
    input_image = Image.open(image_path).convert("RGB")
    transformed_image = img_tr(input_image)

    if align:
        import os
        import dlib
        from alignment import align_face
        # if 'shape_predictor_68_face_landmarks.dat' not in os.listdir():
        #     !wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        #     !bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath='../DiCaprio.jpg', predictor=predictor) 
        input_image = aligned_image
        
    with torch.no_grad():
        latent = encoder(transformed_image.unsqueeze(0).to(device).float())

    return (latent + latent_avg.repeat(latent.shape[0], 1, 1)).detach()