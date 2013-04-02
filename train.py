import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
import csv
from time import time

def feature_extractor():
    features = [
                ('A type', 'A type', f.SimpleTransform()),
                ('B type', 'B type', f.SimpleTransform()),
                ('Bin-Bin', 'Bin-Bin', f.SimpleTransform()),
                ('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
#                ('Ratio of Unique Samples', ['A','B'], f.MultiColumnTransform(f.ratio_unique)),
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                ('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                # uses scipy.special.psi
                #  y=psi(z) is the derivative of the logarithm of the gamma function evaluated at z (also called the digamma function).
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                # scipy.stats.stats.pearsonr
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                # abs(correlation(x,y))
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference))]
                # normalized_entropy(x) - normalized_entropy(y)
    combined = f.FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()

    steps = [("extract_features", features),
#             ("classify",GradientBoostingRegressor(n_estimators=75,
#                                                random_state = 1,
#                                                subsample = .8))] 
             ("classify", RandomForestRegressor(n_estimators=75, # sample code is 50
                                                verbose=0,
                                                n_jobs=-1,
                                                min_samples_split=5,  # sample code is 10
                                                random_state=1,
                                                compute_importances=True,
                                                oob_score=True))]
    return Pipeline(steps)


def get_types(data):
    data['Bin-Bin'] = (data['A type']=='Binary')&(data['B type']=='Binary')
    data['Num-Num'] = (data['A type']=='Numerical')&(data['B type']=='Numerical')
    data['Cat-Cat'] = (data['A type']=='Categorical')&(data['B type']=='Categorical')
    data['Cat-Bin'] = ((data['A type']=='Categorical')&(data['B type']=='Binary')) | \
                    ((data['B type']=='Categorical')&(data['A type']=='Binary'))
    data['Cat-Num'] = ((data['A type']=='Categorical')&(data['B type']=='Numerical')) | \
                     ((data['B type']=='Categorical')&(data['A type']=='Numerical'))
    data['Num-Bin'] = ((data['A type']=='Numerical')&(data['B type']=='Binary')) | \
                     ((data['B type']=='Numerical')&(data['A type']=='Binary'))

    data[['A type','B type']] = data[['A type','B type']].replace('Binary',1)
    data[['A type','B type']] = data[['A type','B type']].replace('Categorical',1)
    data[['A type','B type']] = data[['A type','B type']].replace('Numerical',0)
    return data
def main():
    t1 = time()
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    train_info = data_io.read_train_info()
    train = pd.concat([train,train_info], axis = 1)
    #make function later
    train = get_types(train)
    target = data_io.read_train_target()
    print train

    print("Extracting features and training model")
    classifier = get_pipeline()
    classifier.fit(train, target.Target)
    
    features = [x[0] for x in classifier.steps[0][1].features ]

    csv_fea = csv.writer(open('features.csv','wb'))
    imp = sorted(zip(features, classifier.steps[1][1].feature_importances_), key=lambda tup: tup[1], reverse=True)
    for fea in imp:
        print fea[0], fea[1]
        csv_fea.writerow([fea[0],fea[1]])

    
    oob_score =  classifier.steps[1][1].oob_score_
    print "oob score:", oob_score
    logger = open("run_log.txt","a")
    logger.write("\n" +str( oob_score) + "\n")

    print("Saving the classifier")
    data_io.save_model(classifier)
   
    print("Predicting the train set")
    train_predict = classifier.predict(train)
    trian_predict = train_predict.flatten()
    data_io.write_submission(train_predict, 'train_set', run = 'train')

    t2 = time()
    t_diff = t2 - t1
    print "Time Taken (min):", round(t_diff/60,1)

if __name__=="__main__":
    main()
