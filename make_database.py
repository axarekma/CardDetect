from scripts.cardart import get_set_cards, CardWrapper
import json
from tqdm.auto import tqdm
from pathlib import Path

# Define sets in Model
sets = [
    "LEA",
    "LEB",
    "2ED",
    "CED",
    "CEI",
    "ARN",
    "ATQ",
    "3ED",
    # 'FBB',
    "LEG",
    # 'SUM',
    # 'PDRC',
    "DRK",
    # 'PHPR',
    "FEM",
    # 'PMEI',
    "4ED",
    # '4BB',
    "ICE",
    "CHR",
    # 'BCHR',
    # 'REN','RIN',
    "HML",
]

if __name__ == "__main__":
    DOWNLOAD_IMAGES_FROM_SCRYFALL = False

    DATA_DIR = Path("data/images")
    database = []
    for setcode in tqdm(sets):
        all_cards = get_set_cards(setcode)
        for c in tqdm(all_cards, leave=False):
            database.append(
                CardWrapper(c, DATA_DIR).dump(
                    "small", download=DOWNLOAD_IMAGES_FROM_SCRYFALL
                )
            )

    # Save to a JSON file
    with open("data/card_database.json", "w") as f:
        json.dump(database, f, indent=4)
