from fastai.vision import *
defaults.device = torch.device('cpu')

def load_model():
    """ Load model from path """
    path_to_model = Path('models')

    # Load learner
    learn = load_learner(path_to_model)

    return learn

# def load_image():
#     # Location of bucket with image
#     path = Path('images')
#     url = path/'bear.jpg'

#     # Download Image
#     img = open_image(url)

#     return img
def predict_image(img):
    learn = load_model()

    # Return Prediction
    pred_class,pred_idx,outputs = learn.predict(img)

    return pred_class