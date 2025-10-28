# TeamAID

## Current DINO Train/Test Alterations & Results

- Do not apply random augmentations on test data
    - Should be uniform unlike train data that can be diversified and randomized

- Improvement between DINO and supervised ResNet50 is trivial as of right now
    - May need to prove significance OR
    - change model loading (if wrong) OR
    - Add unlabeled skin cancer data to DINO pretraining

### CURRENT RESULTS on HPC:

#### Supervised ResNet:

Original Test Set Results:
Test Loss: 0.327120
Accuracy: 91.5%
F1 Score: 0.9150
Recall: 0.9152

Job finished at: Tue Oct 28 15:43:59 EDT 2025
======================================================


#### DINO:

Original Test Set Results:
Test Loss: 0.344483
Accuracy: 91.1%
F1 Score: 0.9104
Recall: 0.9106

Job finished at: Tue Oct 28 15:50:24 EDT 2025
======================================================