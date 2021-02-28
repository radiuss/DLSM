# DLSM: Efficient Discrimination and Localization of Multimodal Remote Sensing Images Using CNN-Based Prediction of Localization Uncertainty #

Implements inference of the DLSM model described in:

    [1] M. Uss, B. Vozel, V. Lukin, and K. Chehdi, "Efficient Discrimination and Localization of Multimodal Remote Sensing Images Using CNN-Based Prediction of Localization Uncertainty," Remote Sensing, vol. 12, p. 703, 2020.

The published version can be found [here](https://www.mdpi.com/2072-4292/12/4/703)

If you use this code please cite [1].

Copyright (c) 2021 Mykhail Uss

## Basic Usage ##

### Setting the environment ###

Python Version : 3.5.4

You'll need to install the dependencies from requirements.txt file, something like the following:

```
pip install -r requirements.txt
```

### Testing ###

For testing DLSM model, specify pretrained model location, path to the test data and processed test data.

```
python test_dlsm.py "../PretrainedModel/DLSM_2019_10_22.h5" --testPath "../TestData/testPairsMultimodal_5000_TI32_RI52.mat" --sotaPath "../TestData/testPairsMultimodal_5000_TI32_RI52_ProcessedShift_bound5.mat" --outDir "out"
```

Evaluation script calculates and displays AUC values, ROC corves, SM value histograms, SM maps with correspondence localization for all compared SMs
In additions, sample code for a pair of RI/TI patches processing with DLSM is provided.

## Differences from the original version ##

The model version directly corresponds to the paper [1]. Test data do not correspond to the paper [1]. Small test set was generated for illustration purposes

## Test set and pretrained model ##

For model testing 10000 pairs of RI and TI patches were generated. Test pairs contains 5000 true and 5000 false correspondences and uniformely cover optical-to-optical, optical-to-radar, radar-to-DEM, optical-to-DEM registration cases. Each test pair contains TI patch of size 32 by 32 px, and RI patch of size 52 by 52 px (+-10 px boundary).
Data are stored in TestData/testPairsMultimodal_5000_TI32_RI52.mat

Each test pair was processed by the following SMs: 'NCC', 'MI', 'SIFT-OCT', 'PC', 'HOG', 'MIND', 'HOPC', 'MIgrad', 'L2Net'. See [1] for references to these SMs.
For each SM search zone is +-5 pixels, that is 11 by 11 px SM maps are stored for each SM. Center of these maps correspond to non-shifted RI/TI pair
Data are stored in TestData/testPairsMultimodal_5000_TI32_RI52_Processed_bound5.mat

We provide model DLSM_2019_10_22.h5 trained as described in [1] in "PretrainedModel" folder of this repository.


### constraints ###

DLSM CNN is designed to estimate similarity of two image patches of size 32 by 32 pixels.
