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

def calculate_global_directions(generator, latent_avg, clip_model, styls, test_step_len=5, batch_size=1, device='cuda'):
    transposed_styls = stylst = [torch.stack([styls[i][j] for i in range(len(styls))]) for j in range(len(styls[0]))] #lat_sz1, dataset_sz, lat_sz2
    test_step_lens = [test_step_len*st.std(0) for st in transposed_styls]
    rel = [torch.zeros(*st_l.shape, 512).cpu() for st_l in styls[0]]

    for num_sample in trange(len(styls)):
        with torch.no_grad():
            batch = [st.to(device) for st in styls[num_sample]]
            img, _ = generator([batch], input_is_stylespace=True, randomize_noise=False)
            img_neut = enc_img(clip_model, img)
            del img
            torch.cuda.empty_cache()

            for i in range(len(batch)):
                for j in range(batch[i].shape[2]):
                    batch[i][0, 0, j, 0, 0] += test_step_lens[i][0, 0, j, 0, 0]
                    img2, _ = generator([batch], input_is_stylespace=True, randomize_noise=False)
                    img_tar = enc_img(clip_model, img2)
                    rel[i][0, 0, j, 0, 0, :] += (img_tar - img_neut).sum(0).cpu()
                    del img_tar
                    torch.cuda.empty_cache()
                    batch[i][0, 0, j, 0, 0] -= test_step_lens[i][0, 0, j, 0, 0]
            del batch
    return rel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Global directions',)
    parser.add_argument('--filename')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--generator', default='./stylegan2-ffhq-config-f.pt')
    parser.add_argument('--dataset', default='./styles_dataset.pt')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--save-rate', default=10, type=int)
    parser.add_argument('--step-size', default=0.1, type=float)
    parser.add_argument('--truncation', default=0.7, type=float)

    args = parser.parse_args()

    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    generator, latent_avg = load_generator(args.generator, device=args.device, stylegan_size=stylegan_size, style_dim=512, n_mlp=8)

    dataset = torch.load(args.dataset)

    global_dirs = calculate_global_directions(generator=generator, latent_avg=latent_avg,
                clip_model=clip_model, test_step_len=args.step_size, styls=dataset,
                batch_size=args.batch_size, device=args.device)

    torch.save(global_dirs, f'{args.filename}.pt')