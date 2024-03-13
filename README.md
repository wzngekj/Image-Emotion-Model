# IMAGE EMOTION RECOGNISER NEURAL NETWORK MODEL
## INTRODUCTION
Built with a pre-trained VGG16 model, the model is further enhanced to classify and predict 1 of 6 emotions: `Anger, Contempt, Disgust, Fear, Happy, Sad, Surprise` to successfully detect emotions on novel images of facial expressions
## SUCCESS RATE
Remarkable efficiency at 87.5%
## DATA SOURCE
`Extended Cohn-Kanade (CK+)` dataset folder contains a rich diversity of images with distinct emotions ranging from `Anger, Contempt, Disgust, Fear, Happy, Sad, Surprise`. Each of these distinct images are classified in their respective folders to ensure proper labelling enabling successful and accurate training and evaluation of the `IEM (Image-Emotion-Model)`
## KEY ELEMENTS
1. `Visual Geometry Group (16) Model`: Used as the foundation and further enhanced to suit the current model's tasking. Utilized the pre-trained weights and its deep architecture consisting of 16 layers and uniform design of 3x3 convolutional filters followed by max-pooling layers to accurately capture intricate image patterns due to its depth and simplicity.
2. `Early Stopping`: Prevention of over-fitting to ensure the optimal weights are maintained and returned.
3. `Custom Dense Hidden Layer Structure`: 1 hidden layer with 1024 hidden nodes to siften out distinct characteristics of images. BatchNormalisation incorporated to ensure swifter convergence.
## REQUIREMENTS
1. `cv2`
2. `tensorflow`
3. `numpy`
4. `sci-kit learn`
5. `keras`
6. `matplotlib`
## DEPLOYMENT
### EMOTIONS.PY
1. ``` git clone https://github.com/wzngekj/Image-Emotion-Model.git ```
2. cd IEM
3. pip install tensorflow numpy matplotlib scikit-learn keras
### DATASET
1. Download the dataset ```https://www.kaggle.com/datasets/shuvoalok/ck-dataset/code``` from kaggle
2. Extract the dataset
3. Create a folder named ```data``` to contain all the 7 folders of distinct emotions downloaded
4. Move the `data` folder into the `IEM` folder which contains `emotions.py`
