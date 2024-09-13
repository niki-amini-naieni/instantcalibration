import glob
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Load DINOv2 model.
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)
transform_image = T.Compose(
    [T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
)


def load_images_from_npy(npy_path: str) -> list:
    """
    Loads images from the npy file at [npy_path] and returns a list of tensors
    that can be used as input to DINOv2.

    [npy_path]: file name of numpy file with images to transform

    Returns: list of transformed images as tensors
    """
    images = np.load(npy_path)

    transformed_imgs = []
    for img in images:
        img = img * 255
        if img.shape[-1] != 3:
            # True if [img] is the uncalibrated uncertainty map.
            img = np.stack((img,) * 3, axis=-1)
        img = Image.fromarray(img.astype("uint8"), "RGB")

        transformed_img = transform_image(img)[:3].unsqueeze(0)

        transformed_imgs.append(transformed_img)
    return transformed_imgs


def compute_embeddings(npy_files: list) -> None:
    """
    Computes DINOv2 embeddings for each image in the npy files and saves them
    as N x 384 dimensional arrays in npy files, where N is the number of images
    for a specific npy file, and 384 is the embedding dimension.

    [npy_files]: list of npy file names for the images to encode with DINOv2

    Results are saved in file locations where the images are with the same
    names + suffix "_embeddings.npy"
    """
    with torch.no_grad():
        for npy_file in tqdm(npy_files):
            imgs = load_images_from_npy(npy_file)
            embeddings_list = []
            for img in imgs:
                embeddings = dinov2_vits14(img.to(device))
                embeddings_np = np.array(embeddings[0].cpu().numpy()).reshape(-1)
                embeddings_list.append(embeddings_np)
            all_embeddings = np.array(embeddings_list)
            np.save(npy_file.replace(".npy", "_embeddings.npy"), all_embeddings)


if __name__ == "__main__":

    # Compute and save the embeddings for the inferred images.
    npy_files = glob.glob("scenes/*/preds.npy")
    compute_embeddings(npy_files)

    # Compute and save the embeddings for the uncalibrated uncertainty maps.
    npy_files = glob.glob("scenes/*/uncal_masks.npy")
    compute_embeddings(npy_files)
