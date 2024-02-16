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
sys.path.append("/scratch/cvlab/home/afan/projects/2DImageTransform")
from datasets.dtu_yao import MVSDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
    for j, sample in tqdm(enumerate(TestImgLoader)):
        with torch.no_grad():
            source_points, target_points, imgs = process_batch_DTU(sample, load_size, output_size, config["n_pixel_sample"], device)
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
    for epoch in range(config["max_epochs"]):
        for i, sample in tqdm(enumerate(TrainImgLoader)):
            step = epoch * config["max_steps_per_epoch"] + i
            optimizer.zero_grad()
            # load_size and output_size are assumed to be in the order (w, h)
            source_points, target_points, imgs = process_batch_DTU(sample, load_size, output_size, config["n_pixel_sample"], device)
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



def process_batch_DTU(data, load_size, output_size, num_sample=10, device=torch.device('cuda')):

    images = data['imgs'][0]

    # The range is [-1,1] for the diffusion model
    images = images * 2 - 1

    depth_maps = data['depth_map'][0]
    geo_masks = data['geo_masks'][0, 0]
    proj_matrices = data['proj_matrices'][0]

    _, _, H, W = images.shape
    scale = np.array([W / load_size[0], H / load_size[1]])[None, :, None]

    images = F.interpolate(images, size=(load_size[1], load_size[0]), mode="bilinear")
    depth_maps = depth_maps[None, None]
    depth_maps = F.interpolate(depth_maps, size=(load_size[1], load_size[0]), mode="bilinear")
    geo_masks = geo_masks.float()[None, None]
    geo_masks = F.interpolate(geo_masks, size=(load_size[1], load_size[0]), mode="bilinear")
    geo_masks = geo_masks > 0.5
    proj_matrices[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / scale

    depth_maps = depth_maps[0,0].numpy()
    geo_masks = geo_masks[0,0].numpy()
    proj_matrices = proj_matrices.numpy()

    pool = np.asarray(geo_masks).nonzero()
    sample = np.random.choice(range(len(pool[0])), num_sample)
    pixel_sample = np.concatenate((pool[1][sample][None, :], pool[0][sample][None, :]), axis=0)
    pixel_sample = np.concatenate((pixel_sample, np.ones((1, num_sample))), axis=0)
    depth_sample = depth_maps[(pool[0][sample], pool[1][sample])]

    extrinsics = proj_matrices[:, 0, ...]
    intrinsics = proj_matrices[:, 1, ...]
    K0 = intrinsics[0, :3, :3]
    K = intrinsics[1, :3, :3]
    relative_extrinsics =  extrinsics[1] @ np.linalg.inv(extrinsics[0])
    R = relative_extrinsics[:3, :3]
    t = relative_extrinsics[:3, 3]
    t = t[:, None]
    pixel_sample_prime = K @ (R @ (np.linalg.inv(K0) @ pixel_sample.astype("float32") * depth_sample[None, :]) + t)
    pixel_sample_prime = pixel_sample_prime / pixel_sample_prime[2]

    # rescale and reorder the points, initially the order is (w, h)
    source_points = pixel_sample[0:2].T # from (3, N) to (N, 2)
    target_points = pixel_sample_prime[0:2].T

    source_points = np.flip(source_points, 1)
    target_points = np.flip(target_points, 1)

    source_points = rescale_points(source_points, load_size, output_size)
    target_points = rescale_points(target_points, load_size, output_size)


    return source_points, target_points, images.to(device)


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

    if config.get("train_path"):
        assert config["batch_size"] == 2
        train_dataset = MVSDataset(config["train_path"], config["train_list"], "train", config["n_views"])
        test_dataset = MVSDataset(config["test_path"], config["test_list"], "test", config["n_views"])
        TrainImgLoader = DataLoader(train_dataset, 1, shuffle=True, num_workers=0, drop_last=True)
        TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0, drop_last=False)
        train(config, diffusion_extractor, aggregation_network, optimizer, TrainImgLoader, TestImgLoader)
    else:
        if config.get("weights_path"):
            aggregation_network.load_state_dict(torch.load(config["weights_path"], map_location="cpu")["aggregation_network"])
        test_dataset = MVSDataset(config["test_path"], config["test_list"], "test", config["n_views"])
        TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0, drop_last=False)
        validate(config, diffusion_extractor, aggregation_network, TestImgLoader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train.yaml")
    args = parser.parse_args()
    main(args)