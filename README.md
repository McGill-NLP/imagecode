# ImageCoDe

[![arxiv](https://img.shields.io/badge/arXiv-2203.15867-b31b1b.svg)](https://arxiv.org/abs/2203.15867)

This repository contains code and data for ImageCoDe: [Image Retrieval from Contextual Descriptions](https://arxiv.org/abs/2203.15867). 

![Example](https://github.com/mcgill-nlp/ImageCoDe/blob/main/example.png?raw=true)

## Data
All collected descriptions for the training and validation set are under [`data/train_data.json`](data/train_data.json) and [`data/valid_data.json`](data/valid_data.json).

Image sets can be downloaded on [Zenodo](https://zenodo.org/record/6518944#.YnLboHWZPUQ) or [GoogleDrive](https://drive.google.com/file/d/1OIKNyU0F9lThbaZZ3Jvm7AlF94n1MzDk/view?usp=sharing) and should be unzipped in `data/`.

You can download from the commandline via:

```
wget https://zenodo.org/record/6518944/files/image-sets.zip
```

Alternatively, you can use [HuggingFace Datasets](https://huggingface.co/datasets/BennoKrojer/ImageCoDe) for working with ImageCoDe.

For ViLBERT experiments, you need to download a pretrained ViLBERT checkpoint from volta [here](https://github.com/e-bug/volta/blob/main/MODELS.md), simply by clicking on ViLBERT in the table. Save the downloaded file as `baselines/vilbert/vilbert-pretrained.bin`.
Since ViLBERT uses image features from Faster R-CNN, you also have to downloaded these for all ImageCoDe images here: [Google Drive link](https://drive.google.com/drive/folders/1Gm22SlCM1V63oZIVS0riqWlySL_g5DJc?usp=sharing). Save the file as `data/rcnn-features36-36.lmdb`.
The same procedure applies for UNITER.

The format for [`data/train_data.json`](data/train_data.json) looks like this:

```json
{
  "MSR-VTT-videoTrainValVideo_video2044-shot1_0": {
    "6": "a mom holding her babies in the middle of the picture, no other image intervenes with the image.",
    "7": "The image is fading between a woman holding a baby and a woman sitting with a red background. The hands of the woman sitting aren't visible."
  },
  "video-storytelling-videochristmas_56Nm66j-i5Q-shot14_2": {
  "..."
  }
}
```
And the images under `data/` have the following structure. Each folder contains 10 images. If the images are video frames, the number X in imgX.jpg indicates the frame number:
```
  .
  ├── MSR-VTT-videoTrainValVideo_video2044-shot1_0
      │   ├── img0.jpg
      │   ├── img7.jpg
      │   ├── ...
  ├── video-storytelling-videochristmas_56Nm66j-i5Q-shot14_2
      │   ├── ...
```

Checkpoints for our CLIP models in the paper can be downloaded [here](https://drive.google.com/drive/folders/1PTvUgtyKAqzUPQLRn8792QQx7G-5zs9V?usp=sharing). For example `CONTRA-...` is used to initialize the CLIP vision and text encoder for our strongest baseline model `TEMP-CONTEXTUAL-...`.
### Leaderboard

Based on this you can train your model and test on the unlabeled test set:
```json
{
  "MSR-VTT-videoTestVideo_video7763-shot2_1": [
    "The team name on shirt is visible without a number, but all letters can be seen for team name.",
    "the player can be seen with him on the left close to the logo on the pitch on the right and can be clearly seen"
  ],
  "...":
  ["..."]
}
```

In order to appear on the leaderboard, please format your results in the following format:
```json
{
  "MSR-VTT-videoTestVideo_video7763-shot2_1": [
    1,
    2
  ],
  "...":
  ["..."]
}
```
Where the example here with "1" and "2" represent image indices ranging from 0 to 9.
You can submit to the leaderboard by sending your test set file (or a download link) to benno.krojer@mila.quebec and we will update the leaderboard quickly (max. 1-2 days).
The leaderboard is maintained on the [project website](https://mcgill-nlp.github.io/imagecode/) and might change its submission procedure at some point.

## Installations

Run [`install.sh`](install.sh) for running CLIP experiments.
For VilBERT follow the [instructions for volta](https://github.com/e-bug/volta#repository-setup). 

## Code

Code for CLIP is under [baselines/clip](https://github.com/BennoKrojer/ImageCoDe/tree/main/baselines/clip) and and code for ViLBERT/UNITER is under [baselines/crossencoders](https://github.com/BennoKrojer/ImageCoDe/tree/main/baselines/crossencoders).

For details commands to run each model variant shown in the paper, have a look at the [README in baselines](https://github.com/BennoKrojer/ImageCoDe/tree/main/baselines).

For example to train the best performing model CLIP+TemporalEmbeddings, run:

```
python3 contextual.py --lr 2e-6 --lr_head 1e-4 -b 36 -m ViT-B/16 --fusion mult -a gelu --logit_scale 1000 --finetuned_checkpoint_path checkpoints/CONTRA_clip_best__36_4e-06_30_1395526.pt --add_input --frozen_clip --positional
```

## Data Analysis

Our manual annotation of various phenomena (negation, nuances, ...) in our validation set can be found under `data/manual_annotation_valid.yaml`

## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you want to cite our paper, please use:
```
@inproceedings{krojer_contextual_2022,
  address = {Online},
  title = {Image Retrieval from Contextual Descriptions},
  booktitle = {Proceedings of the 60th {Annual} {Meeting} of the {Association} for {Computational} {Linguistics},
  publisher = {Association for Computational Linguistics},
  author = {Krojer, Benno and Adlakha, Vaibhav and Vineet, Vibhav and Goyal, Yash and Ponti, Edoardo and Reddy, Siva},
  month = may,
  year = {2022},
}
```

## Acknowledgement
Our data (specifically the image sets) are built upon 3 video dataset and Open Images:
- [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)
- [Video Storytelling](https://zenodo.org/record/2383739#.Yizc2Iz0nUR)
- [YouCook](https://web.eecs.umich.edu/~jjcorso/r/youcook/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)

We also the [volta repository](https://github.com/e-bug/volta) for ViLBERT and UNITER baseline variants

For questions or feedback, don't hesitate to contact the author: benno.krojer@mila.quebec

