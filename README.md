# TeamAID

## Current DINO Train/Test Alterations & Results

- Do not apply random augmentations on test data
    - Should be uniform unlike train data that can be diversified and randomized

- Improvement between DINO and supervised ResNet50 is trivial as of right now
    - May need to prove significance OR
    - change model loading (if wrong) OR
    - Add unlabeled skin cancer data to DINO pretraining

### CURRENT RESULTS on HPC:

Supervised ResNet:

Original Test Set Results:
Test Loss: 0.372517
Accuracy: 92.3%
F1 Score: 0.9229
Recall: 0.9227

Job finished at: Wed Oct 22 22:02:27 EDT 2025
Exit code: 
======================================================


DINO:

Original Test Set Results:
Test Loss: 0.365619
Accuracy: 92.6%
F1 Score: 0.9259
Recall: 0.9258

Job finished at: Wed Oct 22 22:07:39 EDT 2025
Exit code: 
======================================================