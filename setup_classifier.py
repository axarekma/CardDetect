import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scripts.cardlookup import display_augmented_images, architectures, Model
from scripts.cardlookup import t_preprocess, t_augment
import torch

with open("data/card_database.json", "r") as f:
    database = json.load(f)


def example_augment():

    images = [el["image"] for el in database]
    display_augmented_images(images, t_augment)
    plt.show()


def train_models():
    n_iters = 200
    model_res = Model(model=architectures.resnet18, database=database)
    model_res.fine_tune(num_epochs=n_iters, lr=0.001, batch_size=64)
    model_res.save("data/model_res.weights")

    model_reg = Model(model=architectures.regnet_y_800mf, database=database)
    model_reg.fine_tune(num_epochs=n_iters, lr=0.001, batch_size=64)
    model_reg.save("data/model_reg.weights")


def test_models():
    model_res = Model(model=architectures.resnet18, database=database)
    model_res.load("data/model_res.weights")
    model_res.calculate_features()

    model_reg = Model(model=architectures.regnet_y_800mf, database=database)
    model_reg.load("data/model_reg.weights")
    model_reg.calculate_features()

    card = database[np.random.randint(len(database))]
    info = f"{card['name']} ({card['set']})"
    for model in [model_res, model_reg]:
        model.test_similarity(Image.open(card["image"]), transform=t_augment, info=info)
    plt.show()


def export_models():
    dummy_input = torch.randn(1, 3, 204, 146)

    model_res = Model(model=architectures.resnet18, database=database)
    model_res.load("data/model_res.weights")

    model_reg = Model(model=architectures.regnet_y_800mf, database=database)
    model_reg.load("data/model_reg.weights")

    torch.onnx.export(
        model_res.model, dummy_input, "data/model_res.onnx", export_params=True
    )
    torch.onnx.export(
        model_reg.model, dummy_input, "data/model_reg.onnx", export_params=True
    )


if __name__ == "__main__":
    # example_augment()  # example of the training data
    # train_models()  # example of the model output
    # test_models()  # example of the model output
    export_models()
