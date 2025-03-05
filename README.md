# Software-VGGNet-Training
This repository is for Part1, Train VGG16 with quantization-aware training, of the ECE 284 VLSI final project.

### File Structure

Software-VGGNet-Training/
├─ code/                                            # Stores all code used in this project
│  ├─ VGG16_Quantization_aware_train.ipynb          # The Jupyter Notebook to train the model
│  ├─ model/                                        # Given in the starter code to use VGG16 model
│  ├─ result/                                       # Store trained VGG16 model checkpoints
│  ├─ *data/                                        # Store downloaded CIFAR10 training data
├─ documentation/                                   # Stores all documentation for this project
├─ README.md                                        # This file, show important information
├─ .git*                                            # Git-related files used for repo configuration

Note: data directory should be ignored in .gitignore file to avoid push issues.

### Project descriptions

- Train for 4-bit input activation and 4-bit weight to achieve >90% accuracy.

- But, this time, reduce a certain convolution layer's input channel numbers to be 8 and output channel numbers to be 8.

- Also, remove the batch normalization layer after the squeezed convolution. e.g., replace "conv -> relu -> batchnorm" with "conv -> relu"

- This layer will be mapped on your 8x8 2D systolic array. Thus, reducing to 8 channels helps your layer's mapping in an array nicely without tiling.

- This time, compute your "psum_recovered" such as HW5 including ReLU and compare with your prehooked input for the next layer (instead of your computed psum_ref).

- [hint] It is recommended not to reduce the input channel of Conv layer at too early layer position because the early layer's feature map size (nij) is large incurring long verification cycles. (recommended location: around 27-th layer, e.g., features[27] for VGGNet)

- Measure of success: accuracy >90%  with 8 input/output channels + error < 10^-3 for psum_recorvered for VGGNet.