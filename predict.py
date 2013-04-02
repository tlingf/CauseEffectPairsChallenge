import data_io
import numpy as np
import pickle
import sys
import pandas as pd

fn  = sys.argv[1]

def historic():
    print("Calculating correlations")
    calculate_pearsonr = lambda row: abs(pearsonr(row["A"], row["B"])[0])
    correlations = valid.apply(calculate_pearsonr, axis=1)
    correlations = np.array(correlations)

    print("Calculating causal relations")
    calculate_causal = lambda row: causal_relation(row["A"], row["B"])
    causal_relations = valid.apply(calculate_causal, axis=1)
    causal_relations = np.array(causal_relations)

    scores = correlations * causal_relations

def main():
    print("Reading the valid pairs") 
    valid = data_io.read_valid_pairs()
    valid_info = data_io.read_valid_info()


    valid[['A type','B type']] = valid[['A type','B type']].replace('Binary',1)
    valid[['A type','B type']] = valid[['A type','B type']].replace('Categorical',1)
    valid[['A type','B type']] = valid[['A type','B type']].replace('Numerical',0)
            
    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions") 
    predictions = classifier.predict(valid)
    predictions = predictions.flatten()

    print("Writing predictions to file")
    data_io.write_submission(predictions, fn)

if __name__=="__main__":
    main()
