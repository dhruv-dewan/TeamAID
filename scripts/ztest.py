import numpy as np
from math import sqrt

# Accuracy values from the image
baseline_accuracy = {
    'A': 40.9 / 100,  # Skin Tone A
    'B': 28.6 / 100,  # Skin Tone B
    'C': 36.2 / 100   # Skin Tone C
}

diverse_accuracy = {
    'A': 62.8 / 100,  # Skin Tone A
    'B': 62.9 / 100,  # Skin Tone B
    'C': 59.4 / 100   # Skin Tone C
}

# Number of samples for each skin tone in the Stanford diverse dataset
n_baseline = 656  # Number of samples for each model in Stanford Diverse dataset
n_diverse = 596

# Function to calculate the Z-score
def z_test(p1, p2, n1, n2):
    # Calculate pooled proportion
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Calculate the standard error
    se = sqrt(p * (1 - p) * (1/n1 + 1/n2))
    
    # Calculate Z-score
    z = (p1 - p2) / se
    return z

# Calculate Z-scores for each skin tone
z_scores = {}
for tone in ['A', 'B', 'C']:
    z_scores[tone] = z_test(baseline_accuracy[tone], diverse_accuracy[tone], n_baseline, n_diverse)

print(z_scores)
