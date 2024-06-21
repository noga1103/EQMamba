# EQMamba: Earthquake Detection and Phase Picking with Mamba

## Overview

EQMamba is an innovative method for simultaneous earthquake detection and seismic phase picking, combining the strengths of the Earthquake Transformer and Mamba models. This project aims to improve the accuracy and efficiency of earthquake monitoring by leveraging deep learning techniques.

## Key Features 

- Simultaneous earthquake detection and phase picking (P and S waves)
- Integration of Mamba layers for efficient processing of long seismic sequences
- Multi-task learning architecture with CNN, BiLSTM, and Mamba blocks
- Data augmentation techniques for improved model robustness
- Trained on the STEAD (STanford EArthquake Dataset) for wide applicability

## Try it yourself 

## Try it yourself

For a complete workflow, you can use the provided [Google Colab notebook](https://colab.research.google.com/drive/1xPcfK0skawQ5xAqkaAWt99sYw5IP7izt?usp=sharing). This notebook will guide you through the process of training and testing the model. The notebook runs on both:

1. A small dataset that can be found in this repository
2. The large STEAD dataset that is imported from Google Drive

If you'd like to download the STEAD dataset yourself, you can find it here: [STEAD Dataset](https://github.com/smousavi05/STEAD)

## Acknowledgments

STEAD dataset (Mousavi et al., 2019)
EQTransformer (Mousavi et al., 2020)
Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu et al., 2023)

## Contact
Noga Bregman - nogabregman@mail.tau.ac.il
Project Link: https://github.com/noga1103/EQMamba
