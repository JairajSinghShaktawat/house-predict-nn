from sklearn import preprocessing
import numpy as np
np.set_printoptions(threshold='nan')  # for array printing purposes


def generate_encoding(features):
    """
    takes in a list of labels and returns a dictionary associating each label
    with a numerical value.
    params: labels - input list of labels to encode
    rtype: dict of labels associated with their numerical value
    """
    features = list(set(features))  # only unique labels
    label_dict = {features[i]: i for i in range(0, len(features))}
    return label_dict


def encode_features(dataset, indices_to_encode=[]):
    """
    takes in a numpy array and a list of indices to encode.
    generates an encoding for each of these values and reassigns the
    categorical variables to numerical variables
    """
    ds = np.array(dataset)
    for index in indices_to_encode:
        feature_dict = generate_encoding(dataset[:, index])
        for i in range(0, dataset.shape[0]):
            ds[i][index] = feature_dict[dataset[i][index]]
    return ds


def isint(x):
    """return true if x is a string representing an int
       params: x - string like
       return: True if x is a string representing an int else False
    """
    for i in x:
        if(i < '0' or i > '9'):
            return False
    return True


def split_labels_from_dataset(dataset, label_idx=-1):
    """
    params: dataset - numpy array of data including labels
            label_idx - index the label is at, for now supports 0 or -1 (last)
    return: dataset: dataset without the labels, and the labels separetely
    """
    if(label_idx == -1):
        labels = dataset[:, -1]
        dataset = dataset[:, 0: -1]
        return dataset, labels
    else:
        labels = dataset[0]
        dataset = dataset[:, 1:]
        return dataset, labels


if __name__ == '__main__':
    fpath_descript = "../data/data_description.csv"
    fpath_train = "../data/house_train.csv"
    fpath_test = "../data/house_test.csv"
    training_data = open(fpath_train).readlines()
    var_names = training_data[0].split(",")
    training_data = training_data[1:]
    dataset = [item.split(",") for item in training_data]
    training_data = np.array(dataset)
    # extract out answers/labels from the array
    training_data, prices = split_labels_from_dataset(training_data,
                                                      label_idx=-1)
    print prices.shape
    print training_data.shape
    encoding = generate_encoding(list(set(training_data[:, 5])))
    encode_these = []
    for i in range(len(training_data[0])):
        if(not isint(training_data[0][i])):
            encode_these.append(i)


    transformed_data = encode_features(training_data, indices_to_encode=encode_these)
    print transformed_data[0]
