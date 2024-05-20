# Task 1

## Structure

To run the scripts, make sure to organize the files as follows:

```
. <--- (project root)
|
+---Furniture_Data
|       
+---Model
    |
    +---Task 1
        |
        +---classifier.py
        |           
        +---models.py
        |           
        +---README.md
        | 
        +---resnet_state_dict.pth
        | 
        +---resnet.pth
        | 
        +---RestNet.ipynb
        | 
        +---vgg16_state_dict.pth
        | 
        +---VGG16.ipynb
        | 
        +---vgg16.pth
```

The trained models (saved as *.pth files) can be found [here](https://rmiteduau.sharepoint.com/:f:/s/COSC2753MachineLearning/EhceSkrgDOpPqfrDC7Yrsf8BO_xriQNQ4woqAIFDNy3S5A?e=kzoJPW).

## Running the classifier

**Prerequisites**

1. Make sure the latest version of Python 3 is installed.

2. Libraries
        
    Due to a bug in the latest torch version (2.3.0), torch and torchvision needs to be downgraded.
    
    Pillow also needs to be installed.
    
    In the project root, run the following command:
    
    `pip install -r "./Model/Task 1/requirements.txt"`

---

To run the classifier, do one of the following:

- Double-click `classifier.py`
- In the project root, execute `python -u "./Model/Task 1/classifier.py"` in the terminal
