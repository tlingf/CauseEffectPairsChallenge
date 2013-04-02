import data_io
import features as f
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
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
             ("classify", RandomForestRegressor(n_estimators=75, # sample code is 50
                                                verbose=2,
                                                n_jobs=-1,
                                                min_samples_split=5,  # sample code is 10
                                                random_state=1))]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.read_train_pairs()
    target = data_io.read_train_target()

    print("Extracting features and training model")
    classifier = get_pipeline()
    classifier.fit(train, target.Target)

    print("Saving the classifier")
    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()
