from sklearn import preprocessing
import numpy as np
np.set_printoptions(threshold='nan') #for array printing purposes




def generate_encoding(labels):
    """
    takes in a list of labels and returns a dictionary associating each label
    with a numerical value.
    params: labels - input list of labels to encode
    rtype: dict of labels associated with their numerical value
    """
    label_dict = {labels[i]: i for i in range(0, len(labels))}
    return label_dict

def encode_labels(dataset, indices_to_encode=[]):
    """
    takes in a numpy array and a list of indices to encode.
    generates an encoding for each of these values and reassigns the
    categorical variables to numerical variables
    """
    for index in indices_to_encode:
        label_dict = generate_encoding(list(set(dataset[:, index])))
        for i in range(0, dataset.shape[0]):
            dataset[i][index] = label_dict[dataset[i][index]]
    return dataset



if __name__ == '__main__':
    fpath_descript = "../data/data_description.csv"
    fpath_train = "../data/house_train.csv"
    fpath_test = "../data/house_test.csv"
    training_data = []
    with open(fpath_train) as file:
        training_data = file.readlines()
    var_names = training_data[0]
    var_names = var_names.split(",")
    training_data = training_data[1:]
    dataset = []
    for item in training_data:
        dataset.append(item.split(","))
    training_data = np.array(dataset)
    encoding = generate_encoding(list(set(training_data[:, 2])))
    print encoding
    transformed_data = encode_labels(training_data, indices_to_encode=[2])
    print transformed_data[0]
    enc = preprocessing.OneHotEncoder()
