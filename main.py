import json
from flask import Flask, request
#from serve import get_keywords_api
# I've commented out the last import because it won't work in kernels, 
# but you should uncomment it when we build our app tomorrow

# create an instance of Flask
app = Flask(__name__)

# load our pre-trained model & function
keywords_api = get_keywords_api()

# Define a post method for our API.
@app.route('/predictimage', methods=['POST'])
def predictimage():
    """ 
    Takes in a json file, predicts image,
    and returns a prediction as a json file.
    """
    # the data the user input, in json format
    input_data = request.json

    # use our API function to get the keywords
    output_data = keywords_api(input_data)

    # convert our dictionary into a .json file
    # (returning a dictionary wouldn't be very
    # helpful for someone querying our API from
    # java; JSON is more flexible/portable)
    response = json.dumps(output_data)

    # return our json file
    return response