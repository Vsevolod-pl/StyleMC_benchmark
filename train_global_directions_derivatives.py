import argparse
import torch
from argparse import Namespace
from tqdm import trange, tqdm

import clip
from StyleGAN import load_generator

stylegan_size=1024
upsample = torch.nn.Upsample(scale_factor=7)
avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

def enc_img(clip_model, img):
    return clip_model.encode_image(avg_pool(upsample(img)))

def enc_lat(generator, clip_model, latents, truncation=0.7, truncation_latent=None):
    img, _ = generator([latents], input_is_latent=True, randomize_noise=False, truncation=0.7, truncation_latent=latent_avg)
    return enc_img(clip_model, img).sum(0)


def calculate_global_directions_der(generator, latent_avg, clip_model, num_epochs=100, batch_size=2, save_name=None, save_rate=None, device='cuda', truncation=0.7, latents=None):
    rel = torch.zeros(18, 512, 512).cpu()
    enc_lat_p = lambda x: enc_lat(generator, clip_model, x, truncation=truncation, truncation_latent=latent_avg)

    if latents is not None:
        num_epochs = len(latents)
    
    for epoch in trange(num_epochs):
        if latents is not None:
            batch = latents[epoch]
        else:
            batch = torch.randn(batch_size, 1, 512).to('cuda')
            batch = batch.expand(batch_size, 18, 512)
        rel_i = torch.autograd.functional.jacobian(enc_lat_p, batch)
        rel_i = rel_i.sum(1)
        rel_i = rel_i.transpose_(0, 1).transpose_(1, 2)
        rel += rel_i.cpu()
        del rel_i
        torch.cuda.empty_cache()
        if save_name is not None and save_rate is not None and epoch % save_rate == save_rate-1:
            torch.save(rel, f'{save_name}_{epoch}.pt')
    return rel


    save_name is not None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Global directions',)
    parser.add_argument('--filename')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--save-rate', default=10, type=int)
    parser.add_argument('--truncation', default=0.7, type=float)

    args = parser.parse_args()

    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=stylegan_size, style_dim=512, n_mlp=8)

    global_dirs = calculate_global_directions_der(generator=generator, latent_avg=latent_avg,
                clip_model=clip_model, num_epochs=args.num_epochs, batch_size=args.batch_size,
                save_name=args.filename, save_rate=args.save_rate, device=args.device,
                truncation=args.truncation)

    torch.save(global_dirs, f'{args.filename}.pt')