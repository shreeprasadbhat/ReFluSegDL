# Retinal Fluid Segmentation of OCT images

https://retouch.grand-challenge.org/

Required modulus

SimpleITK
pyyaml
Tensorflow 2.7
python3 3.7
matplotlib
pandas
numpy
scikit-learn
scipy
tensorflow-addons

# install required libraries
!pip install SimpleITK

!pip install -q pyyaml h5py  # Required to save models in HDF5 format

!pip install -q -U tensorflow-addons

!pip install git+https://github.com/artemmavrin/focal-loss.git

[Network Architecture](unet.png)
