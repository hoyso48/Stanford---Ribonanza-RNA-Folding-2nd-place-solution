## Stanford - Ribonanza RNA Folding 2nd place solution

2nd-place solution to the [Stanford - Ribonanza RNA Folding competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) on kaggle. 

see https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460316 for more details

**Keypoints:**

- Squeezeformer + GRU head.
- Simple Conv2DNet for bpp, adding it as a bias to the attention matrix.
- AliBi positional encoding[1] for robust generalization on longer sequences.
- Weighted loss with signal_to_noise, with longer epochs.
- Additional features for minor score improvements.

The most crucial part of my solution is how to utilize the bpp matrix. I applied a simple shallow Conv2DNet to bpp and directly added it to the attention matrix.

**Features:**

I used some features found useful in the OpenVaccine Challenge, to help fast initial convergence. These included:

- CapR looptype.
- eternafold mfe.
- predicted Looptype with eternafold mfe.
- bpp features (sum, nzero, max).

these features only marginally helped (about -0.0005). Therefore, I believe these features should be removed in the future for the simplicity.

