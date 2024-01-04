## Stanford - Ribonanza RNA Folding 2nd place solution

2nd-place solution to the [Stanford - Ribonanza RNA Folding competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) on kaggle. 

see https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460316 for more details.

**Keypoints:**

- Squeezeformer[1] + GRU head.
- Simple Conv2DNet for bpp, adding it as a bias to the attention matrix.
- AliBi positional encoding[2] for robust generalization on longer sequences.
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

![](model_architecture.png)

## Preparations
1. install required packages
```
git clone https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution
cd Stanford---Ribonanza-RNA-Folding-2nd-place-solution
pip install -r requirements.txt
```
2. download competition dataset.
If you have set up [kaggle-api](https://github.com/Kaggle/kaggle-api), use following commands.
```
mkdir datamount
kaggle competitions download -c stanford-ribonanza-rna-folding -p ./datamount
unzip ./datamount/stanford-ribonanza-rna-folding.zip 
```
Or manually download it https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data in ./datamount and unzip it.
3. prepare preprocessed dataset.
If you have set up [kaggle-api](https://github.com/Kaggle/kaggle-api), use following commands.
```
kaggle datasets download -d hoyso48/stanford-ribonanza-rna-folding-dataset -p ./datamount
unzip ./datamount/stanford-ribonanza-rna-folding-dataset.zip 
```
