from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s :%(levelname)s : %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename =os.path.join(log_dir,"AND_log.log") ,level=logging.INFO, format=logging_str, filemode="a")
logging.basicConfig(level=logging.INFO, format=logging_str)

def main(data, eta, epochs, filename, plotfilename):

    df = pd.DataFrame(data)
    logging.info(f"This is AND DataFrame {df}")
    
    X, y = prepare_data(data)

    model = Perceptron(eta= eta, epochs = epochs)
    model.fit(X,y)

    _ = model.total_loss()  # _ is dummy variable

    save_model(model, filename=filename)
    save_plot(df, plotfilename ,model)

if __name__ == "__main__":

    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }

    df = pd.DataFrame(AND)
    ETA = 0.3 # Learning rate is in between 0 and 1
    EPOCHS = 10
    try :
        logging.info("***** Training has been started*****")
        main(data=df, eta=ETA, epochs=EPOCHS, filename="and.model", plotfilename="and.png") 
        logging.info("***** Training has been completed*****\n") 

    except Exception as e:
        logging.exception(e)
        raise e  