# Use the trained model

# Train the model
## 0. Open up a virtual environment
This can be done anyway you wish. Make sure you also have a terminal/cmd ready with the virtual environment interpreter.
## 1. Install All Dependencies
Ensure you have all the necessary dependencies installed. Run the following command (preferrably in a venv):

``` bash
pip install -r requirements.txt
```

## 2. Create credentials.py file
Create a credentials.py file in the root directory of your project and save your Roboflow API key:

``` python
# credentials.py
ROBOFLOW_APIKEY = "your_roboflow_api_key_here"
```

## 3. Download the data set
Run the dataset_downloader.py program to download the dataset from roboflow. You may modify dataset_downloader.py accordingly if you want to try out with a different dataset.


## 4. Modify shoe_extractor_trainer.py for Non-GPU Systems
If you do not have a GPU, comment out the line that moves the model to CUDA in shoe_extractor_trainer.py:
``` python
# shoe_extractor_trainer.py

# Comment out this line if you don't have a GPU
model.to("cuda")
```
if you have a gpu, and if you are getting erros, run cuda_availability_checker.py to check whehter CUDA is ready to be used. 

