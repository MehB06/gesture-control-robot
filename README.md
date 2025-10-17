# Gesture Control Robot
A gesture-controlled robot program that utilises computer vision to recognise ASL-Alphabet signs and translates them into a mapped action.

This project focuses on the use of **computer vision with pytorch** and **robotic simulation with ROS2**

# COLLABORATERS
- **Fadi Mohamed Mostefai**  
  [![GitHub](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/Fadi-Mostefai)
  [![Portfolio](https://img.shields.io/badge/Portfolio-4CAF50?logo=firefox&logoColor=white)](https://fadi-mostefai.github.io/)

- **Aldrich Antonio Fernandes**  
  [![GitHub](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/Aldrich-Fernandes)
  [![Portfolio](https://img.shields.io/badge/Portfolio-4CAF50?logo=firefox&logoColor=white)](https://aldrich-fernandes.github.io/)

- **Mehdi Belhad**  
  [![GitHub](https://img.shields.io/badge/GitHub-000?logo=github&logoColor=white)](https://github.com/MehB06)
  [![Portfolio](https://img.shields.io/badge/Portfolio-4CAF50?logo=firefox&logoColor=white)](https://mehb06.github.io)


## Features
- [x] Real-time ASL alphabeth detection
- [x] Modular architecture for easier development, debugging and collaboration.
- [ ] Create and train model for segmentation of hand to improve identifiation. 
- [ ] A ROS2 based Gazebo model for action visualisation of the robot
- [ ] Ability to adapt and retrain the model for testing and experimention.
- [ ] Ability to remap ASL inputs to ROS2 actions 

## Technologies
 - Python
 - Pytorch
 - ROS2
 - Git
 - Bash (shell scipting)

## Setup
### Prequisites:
 - Linux (for utilisation with ROS)
 - Working Webcam

### Basic: 
1. Fork repo
2. Clone Repo
  ```bash
  git clone <your-forked-repository-url>
  ```
3. Create venv and source
  ```bash
  python3 -m venv venv      # Creates the virtual environment (venv)
  source venv/bin/activate  # Activates the venv
  ```
4. (inside venv) Import from requirements.txt
  ```bash
  python3 -m pip install --upgrade pip  # Updates 
  pip install -r requirements.txt       # Installs relevent imports

  ```
5. To run the Program, within the project root folder:
  ```bash
  python3 main.py # Runs the program (ensure your inside the venv and in the root dir.)
  ```

### Additional: 
1. Follow setup as described in Basic
2. Follow steps in Dataset to install relevent data
3. Setup Cuda for PyTorch by running the command via this link, following its instructions
  [Pytorch website](https://pytorch.org/get-started/locally/)

## Dataset

If you wish to setup the Dataset, inorder to retrain or play around with the model, do the following:

### OPTION 1: AUTOMATED
Within the root directory of your project folder run the following:
```bash
chmod +x setupDataset.sh  # Ensures the file can run 
./setupDataset.sh         # Runs the file (this may take a minute or 2)
```

The script will automatically download, extract, and organize the ASL Alphabet dataset into the `data/` directory.

### OPTION 2: MANUAL
1. Follow the citation below for the dataset repository within Kaggle.
2. Download the zip file containing the dataset and unzip into your project directory
3. Move the contents of 'asl_alphabeth_train/asl_alphabeth_train/' into a 'data/raw/' directory
    - These are the 29 class folders ('A' to 'Z' + 'del', 'nothing' and 'space') as described in the decription of the page.
4. Delete the .zip and remaining extracted files

## Citations
### DataSet
```bibtex
@misc{nagaraj2018asl,
  title={ASL Alphabet},
  author={Nagaraj, Akash},
  year={2018},
  url={https://www.kaggle.com/datasets/grassknoted/asl-alphabet},
  doi={10.34740/KAGGLE/DSV/29550},
  publisher={Kaggle}
}
```