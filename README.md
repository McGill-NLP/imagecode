# ImageCoDe
This repository contains code and data for ImageCoDe: [Image Retrieval from Contextual Descriptions](https://openreview.net/forum?id=ObaBB7DvM4).

For questions or feedback, don't hesitate to contact the author: benno.krojer@mila.quebec

![Example](https://github.com/BennoKrojer/ImageCoDe/blob/main/example.png?raw=true)

## Data
All collected descriptions for the training and validation set are under [`data/train_data.json`](data/train_data.json) and [`data/valid_data.json`](data/valid_data.json).

Image sets for training and validation can be downloaded [here](https://drive.google.com/file/d/1DpZHH_Y-YstTAJD3ggRZEZ-15kLQohCV/view?usp=sharing) and should be unzipped in `data/`.

For ViLBERT experiments, you need to download pretrained ViLBERT from volta [here](https://github.com/e-bug/volta/blob/main/MODELS.md), simply clicking on ViLBERT in the table. Save the downloaded file as `baselines/vilbert/vilbert-pretrained.bin`.
Since ViLBERT uses image features from Faster R-CNN, you also have to downloaded these for all ImageCoDe images here: [Google Drive link](https://drive.google.com/drive/folders/1Gm22SlCM1V63oZIVS0riqWlySL_g5DJc?usp=sharing). Save the file as `data/rcnn-features36-36.lmdb`.
The same procedure applies for UNITER.

## Installations

Run [`install.sh`](install.sh) for running CLIP experiments.
For VilBERT follow the [instructions for volta](https://github.com/e-bug/volta#repository-setup). 

## Code

Code for CLIP is under [baselines/clip](https://github.com/BennoKrojer/ImageCoDe/tree/main/baselines/clip) and and code for ViLBERT/UNITER is under [baselines/crossencoders](https://github.com/BennoKrojer/ImageCoDe/tree/main/baselines/crossencoders).

For details commands to run each model variant shown in the paper, have a look at the [README in baselines](https://github.com/BennoKrojer/ImageCoDe/tree/main/baselines).

For example to train the best performing model CLIP+TemporalEmbeddings, run:

`python3 contextual.py --lr 2e-6 --lr_head 1e-4 -b 36 -m ViT-B/16 --fusion mult -a gelu --logit_scale 1000 --finetuned_checkpoint_path checkpoints/CONTRA_clip_best__36_4e-06_30_1395526.pt --add_input --frozen_clip --positional`

## Data Analysis

Our manual annotation of various phenomena (negation, nuances, ...) in our validation set can be found under `data/manual_annotation_valid.yaml`

## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you want to cite our paper, please use:
```
To appear soon.
```

## Acknowledgement
Our data (specifically the image sets) are built upon 3 video dataset and Open Images:
- [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)
- [Video Storytelling](https://zenodo.org/record/2383739#.Yizc2Iz0nUR)
- [YouCook](https://web.eecs.umich.edu/~jjcorso/r/youcook/)
- [Open Images](https://storage.googleapis.com/openimages/web/index.html)

We also the [volta repository](https://github.com/e-bug/volta) for ViLBERT and UNITER baseline variants
