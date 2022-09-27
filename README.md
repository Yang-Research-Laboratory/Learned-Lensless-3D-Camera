# Learned-Lensless-3D-Camera
Open source code and demonstrations of learned lensless 3D camera
## [Paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-30-19-34479&id=499281)
![Diagram of learned lensless 3D camera](https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera/blob/main/imgs/Picutre1.PNG)
### Clone this repository:
```
git clone https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera.git
```
## Pre-trained models
We provide multiple pre-trained reconstruction modules (reconM) and enhancement module (enhanceM) for 3D imaging demonstration. The enhanceM is pre-trained from shared [database](https://drive.google.com/drive/u/1/folders/1zS1xuJEx7qU3Qz_h6VH2IJgbrndXQnnq). <br />
The pre-trained models can be found in shared models [folder](https://drive.google.com/drive/u/1/folders/1RIpGIw8NCxSEdc4LDLrfq8EcpbfuuXk3), the models reconM contains 14 physics-aware models trained at object distances from 10-60cm. <br /><br />
![Diagram of learned lensless 3D camera](https://github.com/Yang-Research-Laboratory/Learned-Lensless-3D-Camera/blob/main/imgs/Picture2.gif)
### Test dataset
We provide several raw measurements of real objects by our 3D camera as test data for demonstrating 3D imaging in shared [folder](https://drive.google.com/drive/folders/1nA3Ni0kyoGCLg0OZdjJnQc6SuGjhszZy?usp=sharing).

### Training your own models
To train the models on your customized imagers, you can generated models from our templates, specify the size and format of input images in our example code. 
