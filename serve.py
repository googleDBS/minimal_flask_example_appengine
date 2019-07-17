from fastai.vision import *

def load_model(folder):
    """ Load model from path """
    path_to_model = Path(folder)

    # Load learner
    learn = load_learner(path_to_model)

    return learn