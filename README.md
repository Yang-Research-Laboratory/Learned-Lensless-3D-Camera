# Learned-Lensless-3D-Camera
Open source code and demonstrations of learned lensless 3D camera
## [Paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-19-34479&id=499281)
![Diagram of learned lensless 3D camera](https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera/blob/main/imgs/Picutre1.PNG)
### Clone this repository:
```
git clone https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera.git
```
## Pre-trained models
We provide multiple pre-trained reconstruction modules (reconM) and enhancement module (enhanceM) for 3D imaging demonstration. The enhanceM is pre-trained from shared [**database**](https://drive.google.com/drive/folders/1djiLB1xNhmS91Wp84c0JWDrDP0xsqu7F?usp=sharing). <br />
The pre-trained models can be found in shared models [**folder**](https://drive.google.com/drive/folders/1USPYhWAjOucKdl8uasTL-FYk9-ot_Uy1?usp=sharing), the models reconM contains 14 physics-aware models trained at object distances from 10-60cm. <br /><br />
![Diagram of learned lensless 3D camera](https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera/blob/main/imgs/Picture2.gif)
### Test dataset
We provide several raw measurements of real objects by our 3D camera as test data for demonstrating 3D imaging in shared [**folder**](https://drive.google.com/drive/folders/1HanggfzdR2QkpMYv4vP3KkyXjCebSUaf?usp=sharing).<br />
To test 3D imaging demonstration, download the test datasets into the datasets folder at same directory, and run the code test_3Dimaging.py.
### Training your own models
To train the models on your customized imagers, you can generated models from our [**templates**](https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera/blob/main/models.py), specify the size and format of input images in our example [**code**](https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera/blob/main/test/training_models.py). 
