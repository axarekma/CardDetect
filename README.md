### Card detection using openCV and pytorch

This is a toy project to familiarize myself with openCV and utilize pytorch models in cpp.

The program tries to detect and identify cards from a camera stream.
![Preview](preview.png)


### Usage
If you want to have everything work correctly, you need to download the images from Scryfall and set up the database. This is done by by setting `DOWNLOAD_IMAGES_FROM_SCRYFALL` variable to `True` and running
```
python make_database.py
```

The precalculated features and database in the repo do match, but no image match preview will be avbailable.

You can change the sets you want to include by changing the `sets.` I trained on English sets (images defaulted to the ones without the missing language banner) until homelands. This is also the ONNX file that is included.

### Card detection
Gathering from what I've seen online the canonical (classic) way seems to be to produce a binary mask by morphological operations and getting squares by grabbing suiotable contours in the image. This was a bit tricky to get to work with the potato I used as a camera and the polygons that it detected needed refinement.

I wanted to do something differently, so the idea here is that I use a probabilistic Hough Transform to get the line elements of the image. We then do a DFS on the lines to find cycles of minimum cost.

The cost is ~ the proportional length needed to 
 - Join 'parallel' lines
 - Extend lines to their crossing point, making a corner. 

### Card matching
The unwarped card is matched to a dictionary of feature vectors via a CNN, and the match ranking is based on the Cosine similarity between the detected image and the database of cards.

### CNN model
The card matching is done by similarity comparison using a pre-trained encoder trained with PyTorch. We leverage the pre-trained models of torchvision and fine-tune them on the custom MTG database. Specifically, we train an auto-encoder using image augmentation to maximise the cosine similarity between modified images and the Oracle input.



