# PAGE-Net
### A biologically interpretable integrative deep learning model that integrates PAthological images and GEnomic data
### PAGE-Net has three phrases:
* Patch-wise pre-trained CNN
* Two-stage Aggeragation
* Integration of aggregated pathological images and genomic data

# Get Started
### Patch-wise pre-trained CNN (see pretrain folder)
1. patch_extraction : it extracts valid patchs from WSI by removing background and stains.
2. Datagenerator.py : it makes the data ready for Keras image data loaders. (As data is large we used keras dataloaders for loading data)
3. PAGE_net_pretrain : code for pretraining and saving the pretrained model.

### Two-stage aggregation (see aggregation folder)
1. Aggergation.py : it generates the aggregated score and saves in csv.
2. DataMatching.py : it splits the aggregated data in train, test, and validation

### Integration of aggregated pathological images and genomic data
Run.py: to train the model with the inputs from train.csv. Hyperparmeters are optimized by grid search automatically with validation.csv. C-index is used to evaluate the model performance with test.csv.

#Package Requirments
###Preprocessing
* PyHistopathology package
* Keras
* Tensorflow
* Keract
