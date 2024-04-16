import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from omegaconf import OmegaConf
import pandas as pd
import torch
from tqdm import tqdm
import wandb

from archs.correspondence_utils import (
    load_image_pair,
    batch_cosine_sim,
    points_to_idxs,
    find_nn_source_correspondences,
    draw_correspondences,
    compute_pck,
    rescale_points
)
from archs.stable_diffusion.resnet import collect_dims
from archs.diffusion_extractor import DiffusionExtractor
from archs.aggregation_network import AggregationNetwork

import sys
sys.path.append("/scratch/cvlab/home/afan/projects/disk")
from disk.data import get_datasets2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import (rearrange, reduce, repeat)

def get_rescale_size(config):
    output_size = (config["output_resolution"], config["output_resolution"])
    if "load_resolution" in config:
        load_size = (config["load_resolution"], config["load_resolution"])
    else:
        load_size = output_size
    return output_size, load_size

def log_aggregation_network(aggregation_network, config):
    mixing_weights = torch.nn.functional.softmax(aggregation_network.mixing_weights)
    num_layers = len(aggregation_network.feature_dims)
    num_timesteps = len(aggregation_network.save_timestep)
    save_timestep = aggregation_network.save_timestep
    if config["diffusion_mode"] == "inversion":
        save_timestep = save_timestep[::-1]
    fig, ax = plt.subplots()
    ax.imshow(mixing_weights.view((num_timesteps, num_layers)).T.detach().cpu().numpy())
    ax.set_ylabel("Layer")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers+1))
    ax.set_xlabel("Timestep")
    ax.set_xticklabels(save_timestep)
    ax.set_xticks(range(num_timesteps))
    wandb.log({f"mixing_weights": plt})

def get_hyperfeats(diffusion_extractor, aggregation_network, imgs):
    with torch.inference_mode():
        with torch.autocast("cuda"):
            feats, _ = diffusion_extractor.forward(imgs)
            b, s, l, w, h = feats.shape
    diffusion_hyperfeats = aggregation_network(feats.float().view((b, -1, w, h)))
    img1_hyperfeats = diffusion_hyperfeats[0][None, ...]
    img2_hyperfeats = diffusion_hyperfeats[1][None, ...]
    return img1_hyperfeats, img2_hyperfeats

def compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size):
    # Assumes hyperfeats are batch_size=1 to avoid complex indexing
    # Compute in both directions for cycle consistency
    source_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img1_hyperfeats, img2_hyperfeats)
    target_logits = aggregation_network.logit_scale.exp() * batch_cosine_sim(img2_hyperfeats, img1_hyperfeats)
    source_idx = torch.from_numpy(points_to_idxs(source_points, output_size)).long().to(source_logits.device)
    target_idx = torch.from_numpy(points_to_idxs(target_points, output_size)).long().to(target_logits.device)
    loss_source = torch.nn.functional.cross_entropy(source_logits[0, source_idx], target_idx)
    loss_target = torch.nn.functional.cross_entropy(target_logits[0, target_idx], source_idx)
    loss = (loss_source + loss_target) / 2
    return loss

def save_model(config, aggregation_network, optimizer, step):
    dict_to_save = {
        "step": step,
        "config": config,
        "aggregation_network": aggregation_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    results_folder = f"{config['results_folder']}/{wandb.run.name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    torch.save(dict_to_save, f"{results_folder}/checkpoint_step_{step}.pt")

def validate(config, diffusion_extractor, aggregation_network, TestImgLoader):
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)

    x_ref, y_ref = torch.meshgrid(torch.arange(0, config["load_resolution"]), torch.arange(0, config["load_resolution"]))
    grid_samples = rearrange(torch.stack([x_ref, y_ref]), 'c H W -> c (H W)')

    for j, sample in tqdm(enumerate(TestImgLoader)):
        with torch.no_grad():
            source_points, target_points, imgs = process_batch_megadepth(sample, load_size, output_size, grid_samples, config["n_pixel_sample"], device)
            img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            img1 = imgs[0].permute(1,2,0).cpu().numpy()
            img2 = imgs[1].permute(1,2,0).cpu().numpy()
            img1_feature = img1_hyperfeats[0, 0:3].clip(0,1).permute(1,2,0).cpu().numpy()
            img2_feature = img2_hyperfeats[0, 0:3].clip(0,1).permute(1,2,0).cpu().numpy()
            all_images = []
            all_images.append(wandb.Image(img1, caption="img1"))
            all_images.append(wandb.Image(img2, caption="img2"))
            all_images.append(wandb.Image(img1_feature, caption="img1_feature"))
            all_images.append(wandb.Image(img2_feature, caption="img2_feature"))
            wandb.log({"images": all_images})
        break
 


def train(config, diffusion_extractor, aggregation_network, optimizer, TrainImgLoader, TestImgLoader):
    device = config.get("device", "cuda")
    output_size, load_size = get_rescale_size(config)
    np.random.seed(0)

    x_ref, y_ref = torch.meshgrid(torch.arange(0, config["load_resolution"]), torch.arange(0, config["load_resolution"]))
    grid_samples = rearrange(torch.stack([x_ref, y_ref]), 'c H W -> c (H W)')

    for epoch in range(config["max_epochs"]):
        for i, sample in tqdm(enumerate(TrainImgLoader)):

            step = epoch * config["max_steps_per_epoch"] + i
            optimizer.zero_grad()

            # load_size and output_size are assumed to be in the order (w, h)
            source_points, target_points, imgs = process_batch_megadepth(sample, load_size, output_size, grid_samples, config["n_pixel_sample"], device)
            if source_points is None:
                continue

            img1_hyperfeats, img2_hyperfeats = get_hyperfeats(diffusion_extractor, aggregation_network, imgs)
            loss = compute_clip_loss(aggregation_network, img1_hyperfeats, img2_hyperfeats, source_points, target_points, output_size)
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()})
            if step > 0 and config["val_every_n_steps"] > 0 and step % config["val_every_n_steps"] == 0:
                with torch.no_grad():
                    log_aggregation_network(aggregation_network, config)
                    save_model(config, aggregation_network, optimizer, step)
                    validate(config, diffusion_extractor, aggregation_network, TestImgLoader)
            if i == config["max_steps_per_epoch"]:
                break

def process_batch_megadepth(data, load_size, output_size, grid_samples, num_sample=10, device=torch.device('cuda')):

    bitmaps, images = data

     # The range is [-1,1] for the diffusion model
    bitmaps = bitmaps * 2 - 1

    # assume batch_size = 1
    image_0 = images[0, 0]
    image_1 = images[0, 1]
    reprojections = image_1.project(image_0.unproject(grid_samples))

    flag = sanity_check(grid_samples, reprojections, image_0, image_1)
    pts_0 = grid_samples[:, flag]
    pts_1 = reprojections[:, flag]

    if pts_0.shape[1] < num_sample:
        return None, None, bitmaps[0, 0:2].to(device)
    
    sample = np.random.choice(range(pts_0.shape[1]), num_sample)
    pixel_sample = pts_0[:, sample]
    pixel_sample_prime = pts_1[:, sample]

    # rescale and reorder the points, initially the order is (w, h)
    source_points = pixel_sample[0:2].T # from (3, N) to (N, 2)
    target_points = pixel_sample_prime[0:2].T

    source_points = np.flip(source_points.numpy(), 1)
    target_points = np.flip(target_points.numpy(), 1)

    source_points = rescale_points(source_points, load_size, output_size)
    target_points = rescale_points(target_points, load_size, output_size)

    return source_points, target_points, bitmaps[0, 0:2].to(device)

def sanity_check(kps1, kps2, img1, img2):
    # reproject to the other image.
    kps1_r = img2.project(img1.unproject(kps1)) # [2, N]
    kps2_r = img1.project(img2.unproject(kps2)) # [2, N]

    # compute pixel-space differences between (kp1, repr(kp2))
    # and (repr(kp1), kp2)
    diff1 = kps2_r - kps1# [2, N]
    diff2 = kps1_r - kps2 # [2, N]

    # NaNs indicate we had no depth available at this location
    has_depth = (torch.isfinite(diff1) & torch.isfinite(diff2)).all(dim=0)

    # threshold the distances
    close1 = torch.norm(diff1, p=2, dim=0) < 2.0
    close2 = torch.norm(diff2, p=2, dim=0) < 2.0

    good_pairs = close1 & close2

    return good_pairs


def load_models(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    device = config.get("device", "cuda")
    diffusion_extractor = DiffusionExtractor(config, device)
    dims = config.get("dims")
    if dims is None:
        dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]
    aggregation_network = AggregationNetwork(
            projection_dim=config["projection_dim"],
            num_norm_groups=config["num_norm_groups"],
            feature_dims=dims,
            device=device,
            save_timestep=config["save_timestep"],
            num_timesteps=config["num_timesteps"]
    )
    return config, diffusion_extractor, aggregation_network

def main(args):
    config, diffusion_extractor, aggregation_network = load_models(args.config_path)

    # wandb.init(project=config["wandb_project"], name=config["wandb_run"])
    wandb.init(project=config["wandb_project"], name=config["wandb_run"], mode=config["wandb_mode"])

    wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
    parameter_groups = [
        {"params": aggregation_network.mixing_weights, "lr": config["lr"]},
        {"params": aggregation_network.bottleneck_layers.parameters(), "lr": config["lr"]}
    ]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])

    train_dataloader, test_dataloader = get_datasets2(
            config["train_path"],
            no_depth=False,
            batch_size=config["megadepth_batch_size"],
            train_limit=config["train_scene_limit"],
            test_limit=config["test_scene_limit"],
            crop_size=(config["load_resolution"], config["load_resolution"]),
        )
    if config.get("train_path"):
        train(config, diffusion_extractor, aggregation_network, optimizer, train_dataloader, test_dataloader)
    else:
        if config.get("weights_path"):
            aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
        validate(config, diffusion_extractor, aggregation_network, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()
    main(args)