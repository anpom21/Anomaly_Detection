# TODO list
- Finish lighttest.py
  - Finish Dataloader (train, val, test)
  - Finsih testFunction
  - Train new models
  - Add plot of accuracy/precision
  - Add training plots
- Train models with x amount of light from different direction to make mean of accuracy of models. (for all x amount of light models).

# Anomaly_Detection
This project tests a simple anomaly detection network based on an auto encoder network. This will be grounds for the first Anomaly Detection model using photometric lighting.


# Dependencies:
This project is setup for Windows and is set to train using GPU. It is recommended to set up a virtual environment for the packages listet below.
You can follow this guide on how to set up a virtual environment in visual studio code: https://code.visualstudio.com/docs/python/environments

AI/Computer vision libraries:
- Pytorch: https://pytorch.org/get-started/locally/
  For windows:  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- OpenCV:       pip install opencv-python
- Anomalib:     pip install anomalib

Standard libraries:
- numpy:      pip install numpy
- PIL:        pip install pillow
- matplotlib: pip install matplotlib
- tqdm:       pip install tqdm
- seaborn:    pip install seaborn
- sklearn:    pip install scikit-learn

Dataset:
- MVTec AD

# Guides:
- https://www.youtube.com/watch?v=MIxnMC0Zv0Y&list=PLoSULBSCtofdd9Lbp_6uDV0Vqet0afri5&ab_channel=IntelligentMachines

#Models to research:
- Auteencoder
- Simplenet
- Patchcore
![paper_tree](https://github.com/user-attachments/assets/f280c43a-8696-4e87-a5ec-ed9d796e3dbf)

