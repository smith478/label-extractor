# Label Extractor
Extract labels from free text radiology reports

## Dataset

A few datasets to experiment with:
- MIMIC-III
- i2b2 2012 NLP

## Approach overview

In order to get our initial set of labels the idea is to use `zero_shot_model.py`. This script uses a generative LLM (e.g. Llama 3) to try to identify presence of a set of abnormalities from the free text within each radiology report. 

We then inspect the results of these quasi-labels to see how trustworthy they are and also clean up the labels. We also verify that they are properly formatted.

Next we will fine tune a classification (with the classes being the abnormalities we are trying to detect) LLM using `fine_tune.py`, and do an error analysis to both look at the quality of the model and also to do another quality check on our labels. This may involve some iterative fine tuning and label cleanup.

## Model training

- Step 1: Use `zero_shot_model.py` to get pseudo labels from llama 3
- Step 2: Run `fine_tune.py` to train a model on the pseudo labels from step 1
- Step 3: Run an error analysis and clean up mis-labeled samples
- Step 4: Repeat steps 2 and 3 until we get decent results

## TODO 

- Add requirements.txt
- Get model training part working with a few different variants/options
