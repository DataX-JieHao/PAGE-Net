# PAthological & GEnomic (PAGE-Net).
PAGE-Net has three phases.
* Pretrained patch-based CNN.
* Two-stage Aggeragation.
* Integration of pathology and genomic data.
Code flow and instrutions for three phases are explained below.

## Pretrained patch-based CNN.
Codes for this phase are in pretrain folder.
1. patch_extraction : it extracts valid patchs from WSI by removing background and stains.
2. Datagenerator.py : it makes the data ready for keras image data loaders. (As data is large we used keras dataloaders for loading data)
3. PAGE_net_pretrain : Code for pretraining and saving the pretrained model.

## TWO-stage Aggeragation.
Codes for this phase are in aggergation folder.
1. Aggergation.py : it generates the aggregated score and saves in csv.
2. DataMatching.py : it splits the aggregated data in train,test, and validation
