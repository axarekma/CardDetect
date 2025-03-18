import torch

# from torchvision.transforms import v2
from torchvision import transforms as v2
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import random
from tqdm.auto import trange
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, OneCycleLR


def top_k_indices(similarities, k=5):
    # Convert the list of similarities to a tensor
    similarities_tensor = torch.tensor(similarities)

    # Get the top k indices and values
    values, indices = torch.topk(similarities_tensor, k)

    return indices.tolist()


class RandomAffineWithRandomFill:
    def __init__(self, degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=2):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, img):
        # Generate a random fill color for this specific image
        random_fill = tuple(random.randint(0, 255) for _ in range(3))

        # Apply RandomAffine with the generated fill color
        transform = v2.RandomAffine(
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            fill=random_fill,
        )

        return transform(img)


def display_augmented_images(images, aug_f):
    fig, subplots = plt.subplots(2, 5, figsize=(13, 6))
    for i in range(5):
        axi1 = subplots.flat[i]
        axi2 = subplots.flat[i + 5]

        original_img = Image.open(images[i])
        augmented_img = aug_f(original_img)

        axi1.imshow(original_img)
        axi2.imshow(augmented_img.permute(1, 2, 0).numpy())
        axi1.set_title("original_img")
        axi2.set_title("augmented_img")


t_augment = v2.Compose(
    [
        v2.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1),
        RandomAffineWithRandomFill(),
        v2.GaussianBlur(kernel_size=11, sigma=(0.1, 3.0)),
        v2.RandomPosterize(bits=4, p=0.1),
        v2.ToTensor(),
    ]
)
t_augment_flip = v2.Compose(
    [
        v2.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1),
        RandomAffineWithRandomFill(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.GaussianBlur(kernel_size=11, sigma=(0.1, 3.0)),
        v2.RandomPosterize(bits=4, p=0.1),
        v2.ToTensor(),
    ]
)
t_preprocess = v2.Compose([v2.ToTensor()])


class BaseModelFactory:
    @property
    def resnet18(self):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    @property
    def efficientnet_b3(self):
        return models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    @property
    def regnet_y_800mf(self):
        return models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)


architectures = BaseModelFactory()


class CustomImageDataset(Dataset):
    def __init__(self, database, preprocess=None, augment=None):
        self.database = database
        self.preprocess = preprocess
        self.augment = augment

        self.lookup_name = {
            name: i for i, name in enumerate(set([el["name"] for el in database]))
        }
        self.lookup_set = {
            name: i for i, name in enumerate(set([el["set"] for el in database]))
        }

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        el = self.database[idx]
        image = Image.open(el["image"])

        return (
            self.lookup_name[el["name"]],
            self.lookup_set[el["set"]],
            self.augment(image),
            self.preprocess(image),
        )


class Model:
    def __init__(self, model, database):
        self.model = model
        self.database = database
        self.features = None

    def __call__(self, img):
        return self.model(img)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def init_dictionary_old(self, images):
        self.model.to("cpu")
        self.images = images
        self.features = [
            self.model(t_preprocess(Image.open(images[ind2])).unsqueeze(0))
            for ind2 in trange(len(images))
        ]

    def calculate_features(self, batch_size=32, device="cuda"):
        self.features = []

        self.model.eval()
        self.model = self.model.to(device)
        # Process in batches
        with torch.no_grad():  # No need to track gradients
            for i in trange(0, len(self.database), batch_size):
                batch_images = [
                    Image.open(el["image"]) for el in self.database[i : i + batch_size]
                ]

                # Load and preprocess images inside the loop
                batch_tensors = [
                    t_preprocess(img).unsqueeze(0) for img in batch_images
                ]  # List of tensors
                batch_tensors = torch.cat(batch_tensors, dim=0).to(
                    device
                )  # Stack and send to GPU

                features = self.model(batch_tensors)  # Forward pass through model
                self.features.append(features)

        self.features = torch.cat(self.features, dim=0).to("cpu")

    def get_similar(self, image, transform=t_preprocess, k=5, rotate=False):
        self.model.eval()
        self.model = self.model.to("cpu")
        image = np.copy(image[::-1, ::-1]) if rotate else image
        t_image = transform(image)
        feature_test = self(t_image.unsqueeze(0))
        similarities = [
            float(F.cosine_similarity(feature_test, feature_ref))
            for feature_ref in self.features
        ]

        topk = top_k_indices(similarities, k)
        images = [
            np.array(Image.open(self.database[ind_match]["image"]))
            for ind_match in topk
        ]
        scores = [similarities[ind_match] for ind_match in topk]
        return images, scores

    def find_similar(self, image, transform=t_preprocess, k=5):
        images, scores = self.get_similar(image, transform, k)
        images_rot, scores_rot = self.get_similar(image, transform, k, rotate=True)

        if np.mean(scores) > np.mean(scores_rot):
            return images, scores
        else:
            return images_rot, scores_rot

    def testbatch(self, batch_size=4):
        dataset = CustomImageDataset(
            self.database, preprocess=t_preprocess, augment=t_augment
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batched_elemenmt in dataloader:
            return batched_elemenmt

    def test_similarity(self, image, transform, k=5, info=None):
        self.model.eval()
        device = next(self.model.parameters()).device  # Detect model device

        # Transform image and move to the correct device
        t_image = transform(image).to(device)

        # Compute test image embedding
        feature_test = F.normalize(self(t_image.unsqueeze(0)), dim=-1)

        # Compute batch-wise cosine similarity for efficiency
        features_ref = F.normalize((self.features).to(device), dim=-1)
        similarities = torch.mm(feature_test, features_ref.T).squeeze(0)

        # Get top-k matches
        top5 = similarities.topk(k).indices.tolist()

        # Visualization
        fig, ax = plt.subplots(ncols=k + 1, figsize=(3 * (k + 1), 3))
        ax[0].imshow(t_image.permute(1, 2, 0).cpu().numpy())
        title = "Query Image" if info is None else info
        ax[0].set_title(title)

        for i, ind_match in enumerate(top5):
            match = self.database[ind_match]
            ax[i + 1].imshow(Image.open(match["image"]))
            ax[i + 1].set_title(
                f"{match['name']} ({match['set']}) {similarities[ind_match]:.4f}",
                fontsize=10,
            )
            ax[i + 1].axis("off")

        plt.show()

    def test_similarity_old(self, image, transform, k=5, info=None):
        device = torch.device("cpu")
        self.model = self.model.to(device)
        t_image = transform(image)
        feature_test = self(t_image.unsqueeze(0))
        similarities = [
            float(F.cosine_similarity(feature_test, feature_ref))
            for feature_ref in self.features
        ]

        top5 = top_k_indices(similarities, k)

        # Visualization
        fig, ax = plt.subplots(ncols=k + 1, figsize=(3 * (k + 1), 3))
        ax[0].imshow(t_image.permute(1, 2, 0).cpu().numpy())
        title = "Query Image" if info is None else info
        ax[0].set_title(title)

        for i, ind_match in enumerate(top5):
            match = self.database[ind_match]
            ax[i + 1].imshow(Image.open(match["image"]))
            ax[i + 1].set_title(
                f"{match['name']} ({match['set']}) {similarities[ind_match]:.4f}",
                fontsize=10,
            )
            ax[i + 1].axis("off")

    def fine_tune(self, num_epochs=10, batch_size=64, lr=0.001, use_scheduler=True):
        # Create the dataset and dataloader
        dataset = CustomImageDataset(
            self.database, preprocess=t_preprocess, augment=t_augment
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        n_batches = len(dataloader)

        # Optimizer (Adam for fine-tuning)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=n_batches * num_epochs)

        # # Device setup (use GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Training loop
        self.model.train()
        t = trange(num_epochs)

        ave_loss = []
        min_diag = []
        max_off = []

        for epoch in t:
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            min_diag_batch = []
            max_off_batch = []
            for names, sets, inputs, oracle in dataloader:
                names, sets, inputs, oracle = (
                    names.to(device),
                    sets.to(device),
                    inputs.to(device),
                    oracle.to(device),
                )

                optimizer.zero_grad()  # Zero the gradients
                # Forward pass
                embed_oracle = self.model(oracle)
                embed_input = self.model(inputs)
                # make cc matrix to do adverse loss

                margin = 0.2
                synonym_score = 0.9
                set_score = 0.2
                # https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572
                x_cosine_similarity = F.cosine_similarity(
                    embed_input[None, :, :], embed_oracle[:, None, :], dim=-1
                )
                pairwise_names = (names[:, None] == names[None, :]).float() * (
                    synonym_score - margin
                )
                pairwise_sets = (sets[:, None] == sets[None, :]).float() * (
                    set_score - margin
                )
                final_margin = torch.max(pairwise_names, pairwise_sets) + margin

                bs_curr = x_cosine_similarity.shape[0]

                embed_loss = torch.max(
                    x_cosine_similarity - final_margin,
                    torch.zeros_like(x_cosine_similarity, device=device),
                )
                embed_loss[range(bs_curr), range(bs_curr)] = (
                    1 - x_cosine_similarity.diag()
                ) * bs_curr
                loss = embed_loss.sum()
                loss.backward()
                optimizer.step()

                # Track loss and accuracy
                running_loss += loss.item()

                # Extract diagonal (same-class similarities)
                if bs_curr == batch_size:
                    # This had issues with e.g. batch sizes of 1
                    # only extract data for full batches

                    diag_values = x_cosine_similarity.diag()
                    min_diag_batch.append(diag_values.min().item())
                    # Extract off-diagonal (different-class similarities)
                    off_diag_mask = 1 - torch.eye(bs_curr, device=device)
                    off_diag_values = x_cosine_similarity[off_diag_mask.bool()]
                    max_off_diag = off_diag_values.max().item()
                    max_off_batch.append(max_off_diag)

                if use_scheduler:
                    scheduler.step()

            ave_loss.append(running_loss / len(dataloader))
            min_diag.append(np.mean(min_diag_batch))
            max_off.append(np.mean(max_off_batch))
            t.set_description(
                f"avg_loss {ave_loss[-1]:.1e} {max_off[-1]:.2f}--{min_diag[-1]:.2f}"
            )

        print("Training complete.")
        self.model = self.model.to("cpu")
        f, (ax1, ax2) = plt.subplots(ncols=2)
        ax1.semilogy(ave_loss, label="Loss")
        ax2.plot(min_diag, label="Min Diagonal")
        ax2.plot(max_off, label="Max Off-Diagonal")
        ax1.legend()
        plt.show()

    def eval(self):
        self.model.eval()
        self.model.to("cpu")
