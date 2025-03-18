import scrython
import itertools
import time
import requests
from pathlib import Path
import imageio
import requests


def ensure_path_exists(path: Path):
    """Ensure that the parent directories of the given path exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def get_scryfall_image(url: str, path: str):
    """
    Download an image file from Scryfall.
    @param url: Url to the image.
    @param path: Path to save the image.
    @return: True if successful, False if failed.
    """
    with requests.get(url) as response:
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        return False


def get_set_cards(set_code):
    set_cards = []
    for page in itertools.count(1, 1):
        time.sleep(0.1)
        try:
            data = scrython.cards.Search(q=f"set:{set_code}", page=page, unique="art")
            for card in data.data():
                set_cards.append(card)
        except:
            break
    print(f"Got {len(set_cards)} cards for {set_code}")
    return set_cards


class CardWrapper:
    def __init__(self, card, folder=None):
        self.card = card
        self.folder = Path.cwd() if folder is None else folder

    def __getitem__(self, key):
        return self.card[key]

    @property
    def name(self):
        return self.card["name"]

    @property
    def id(self):
        return self.card["id"]

    @property
    def set(self):
        return self.card["set"]

    def image_path(self, version):
        return self.folder / str(version) / f"{self.id}.jpg"

    def dump(self, version, download=False):
        # ensure_art
        if download:
            self.download_image(version)
        return {
            "name": self.name,
            "id": self.id,
            "set": self.set,
            "image": str(self.image_path(version).resolve()),
        }

    def download_image(self, version):
        url = self.card["image_uris"][version]
        file_path = self.image_path(version)
        ensure_path_exists(file_path)
        if not file_path.exists():
            # print('Downloading from Scryfall')
            get_scryfall_image(url, file_path)
            time.sleep(0.1)

    def get_image(self, version):
        self.download_image(version)
        return imageio.imread(self.image_path(version))  # Return existing image path
