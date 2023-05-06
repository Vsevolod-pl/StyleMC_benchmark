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

def calculate_global_directions(generator, latent_avg, clip_model, latents, test_step_len=5, batch_size=1, device='cuda'):
    rel = torch.zeros(18, 512, 512).cpu()
    dlat = torch.zeros(1, 18, 512).to(device)
    test_step_lens = test_step_len*latents.std(0)[0]

    for num_sample in trange(len(latents)):
        with torch.no_grad():
            batch = latents[num_sample].to(device)
            img, batch, _ = generator([batch], input_is_latent=False, randomize_noise=False, truncation=0.7, truncation_latent=latent_avg, return_latents=True)
            img_neut = enc_img(clip_model, img)
            del img
            torch.cuda.empty_cache()

            for j in range(512):
                for i in range(18):
                    dlat *= 0
                    dlat[0, i, j] = test_step_lens[i, j]
                    img2, _ = generator([batch + dlat], input_is_latent=True, randomize_noise=False)
                    img_tar = enc_img(clip_model, img2)
                    rel[i, j] += (img_tar - img_neut).sum(0).cpu()
                    del img_tar
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
    parser.add_argument('--dataset', default='./latents_dataset.pt')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--save-rate', default=10, type=int)
    parser.add_argument('--step-size', default=0.1, type=float)
    parser.add_argument('--truncation', default=0.7, type=float)

    args = parser.parse_args()

    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=stylegan_size, style_dim=512, n_mlp=8)

    dataset = torch.load(args.dataset)

    global_dirs = calculate_global_directions(generator=generator, latent_avg=latent_avg,
                clip_model=clip_model, test_step_len=args.step_size, latents=dataset,
                batch_size=args.batch_size, device=args.device)

    torch.save(global_dirs, f'{args.filename}.pt')