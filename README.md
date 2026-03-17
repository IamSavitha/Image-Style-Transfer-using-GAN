# CycleGAN Image Style Transfer 

This project implements a **CycleGAN (Cycle-Consistent Generative Adversarial Network)** from scratch to perform **unpaired image-to-image translation** between Monet paintings and real-world photographs.

---

## Project Overview

The goal of this project is to learn a mapping between two visual domains **without paired training data**:

- Monet Paintings → Real Photos  
- Real Photos → Monet Paintings  

Using CycleGAN, the model learns both forward and backward mappings while preserving content through **cycle-consistency loss**.

---

## Objectives

- Implement **CycleGAN from scratch**
- Perform **unpaired image translation**
- Understand **GAN training dynamics**
- Learn **cycle-consistency constraints**
- Generate visually convincing style transfers

---

##  Model Architecture

The CycleGAN model consists of:

### Generators
- **G (Monet → Photo)**  
- **F (Photo → Monet)**  

Both generators use:
- ResNet-based architecture  
- Downsampling → Residual Blocks → Upsampling  
- Instance Normalization  

---

### Discriminators

- **D_A (Monet domain)**  
- **D_B (Photo domain)**  

Each uses:
- PatchGAN architecture  
- 30×30 patch-level classification  
- Spectral normalization  

---

### Cycle Consistency

The model enforces:
A → B → A
B → A → B


Ensuring content is preserved while style changes.

---

## Training Pipeline

### Forward Cycle
real_A → G → fake_B → F → rec_A


### Backward Cycle

real_B → F → fake_A → G → rec_B


---

## ⚙️ Loss Functions

The total generator loss consists of:

### 1. Adversarial Loss (LSGAN)
- Encourages realistic image generation  
- Uses Mean Squared Error (MSELoss)

### 2. Cycle Consistency Loss (λ = 10)
L1(rec_A, real_A) + L1(rec_B, real_B)


### 3. Identity Loss (λ = 5)
L1(G(B), B) + L1(F(A), A)


### Final Loss
Loss_G = Adversarial + Cycle + Identity


---

## ⚙️ Training Setup

- Optimizer: **Adam**
- Learning Rate: `2e-4`
- β₁: `0.5`
- Epochs: `140`
- Batch Size: `1`
- Image Size: `256 × 256`

### Training Tricks

- Replay Buffer (size = 50) for discriminator stability  
- Linear LR decay after 100 epochs  
- Mixed precision training (AMP)  

---

## Results

### Monet → Photo
- Converts paintings into realistic textures  
- Preserves scene structure  

### Photo → Monet
- Applies impressionist color palette  
- Adds painterly brush strokes  

---

##  Sample Outputs

| Input | Output |
|------|--------|
| Monet Painting | Photorealistic Image |
| Real Photo | Monet-style Painting |

*(Add images here in your repo for better visualization)*

---

##  Strengths

- Works without paired datasets  
- Preserves content through cycle consistency  
- Generates high-quality stylistic transformations  
- Stable training with replay buffer  

---

##  Limitations

- Computationally expensive training  
- Limited resolution (256×256)  
- Possible geometric distortions  
- Sensitive to hyperparameters  

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- Matplotlib  

---

## Project Structure

CycleGAN/
│
├── data/
├── models/
│ ├── generator.py
│ ├── discriminator.py
│
├── train.py
├── utils.py
├── losses.py
└── README.md


---

## How to Run

### 1. Install Dependencies

pip install torch torchvision matplotlib


### 2. Train Model
python train.py


---

## References

- Zhu et al., *CycleGAN: Unpaired Image-to-Image Translation* (2017)  
- GANs (Goodfellow et al., 2014)  

---

## Key Learnings

- Understanding adversarial training  
- Balancing generator vs discriminator  
- Importance of cycle-consistency  
- Stabilization techniques in GANs  

---

##  Future Improvements

- Train on higher resolution images  
- Use perceptual loss for better quality  
- Add attention mechanisms  
- Optimize training time  

---

## Author

**Savitha Vijayarangan**  
**Keith Rajesh Gonslave**# Image-Style-Transfer-using-GAN
# Image-Style-Transfer-using-GAN
