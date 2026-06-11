# A Unified Hand and Gesture Tracking via Offloading Framework for Object-mediated Interaction in Wearable AR - UnifiedHOInteraction

Woojin Cho, Taewook Ha, Taejun Son, Woontack Woo, VR2026


## Overview
UnifiedHOInteraction provides the following folders and models:

- `Device`: Device-side Unity project for Hololens 2
- `Server`: Server-side Python project


## Installation

- This code is tested with PyTorch 2.0.0 and Python 3.10.18 on Linux and Windows 11.
- Clone and install the following main packages.
    ```bash
    git clone git@github.com:UVR-WJCHO/UnifiedHOInteraction.git
    cd UnifiedHOInteraction
    pip install -r requirements.txt
    ```
	

## Download pretrained models
1. Download the pretrained model (zip file) from the link below:   https://www.dropbox.com/scl/fi/6z7d6x3mscsh5hsqa4tqn/UnifiedHOInteraction_models.zip?rlkey=zfg29fvkmpd57isg1y5msuvj6&st=5j2qbm75&dl=0
   
2. Unzip the downloaded file into the following directory:
   `UnifiedHOInteraction/Server`

3. After extraction, the folder structure should look like this:
   ```
   UnifiedHOInteraction/
   └── Server/
       └── handtracker_wilor/
           ├── pretrained_models/
               ├── wilor_final.ckpt
       └── handtracker/
           ├── checkpoint/
               ├── SAR_AGCN4_cross_wBGaug_extraTrue_resnet34_s0_Epochs50.ckpt
			       ├── checkpoint.pth
       └── gestureclassifier/
           ├── checkpoints/
               ├── checkopint.tar
	```


## Run

- The Server PC and Device must be connected to the **same network** for communication.

### Device

- Enable **Developer Mode** on the Hololens 2.
- Record the device's **Wi-Fi IP address**.
- Build and deploy the Unity application onto the Hololens 2.
- **Tested Environment:** Tested on **Unity version 2022.3.60f1** with the following setup: [Specific setup details TBD, e.g., MRTK version, build target].


### Server

- Update the `HOST_ADDRESS` variable in `main_on_hl2.py` with the Device's recorded **IP address**.
- **Run the device application first**, then execute the following server module:
    ```bash
    cd server
    python main_on_hl2.py
    ```



## Acknowledgement
- This work was supported by Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2019-II191270, WISE AR UI/UX Platform Development for Smartglasses)

- The sensing data acquisition using HoloLens 2 Research Mode was implemented with reference to [project](https://github.com/jdibenes/hl2ss/).


## Lisense
WiseUI Applications are released under a MIT license. 
For a closed-source version of WiseUI's modules for commercial purposes, please contact the authors : uvrlab@kaist.ac.kr, woojin.cho@kaist.ac.kr

