from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotfilename):

    X, y = prepare_data(data)

    model = Perceptron(eta= eta, epochs = epochs)
    model.fit(X,y)

    _ = model.total_loss()

    save_model(model, filename=filename)
    save_plot(df, plotfilename ,model)

if __name__ == "__main__":

    OR = {
        "x1": [0,1,0,1],
        "x2": [0,0,1,1],
        "y": [0,1,1,1]

    }

    df = pd.DataFrame(OR)
    ETA = 0.3 # Learning rate is in between 0 and 1
    EPOCHS = 10

    main(data=df, eta=ETA, epochs=EPOCHS, filename="or.model", plotfilename="or.png")    