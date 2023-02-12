from FeatureGenerator import rate_transform
import pickle

# load data
filename = "./featuresets/v1_1/feature_data_v1_1.pickle"
with open(filename,'rb') as f:
    data = pickle.load(f)
    f.close()

# transform
data = rate_transform(data)

# Save
out_file = "./featuresets/V1_2/feature_data_v1_2.pickle"
with open(out_file, "wb") as f:
    pickle.dump(data, f)
    f.close()