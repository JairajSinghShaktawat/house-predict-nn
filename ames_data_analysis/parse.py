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


def get_categorical_indices(dataset):
    """
    given a dataset, finds all indices j such that dataset[i][j] is a
    not a continuous integer variable, for any i. Use for encoding categorical
    variables.
    params: dataset - the dataset with possible categorical variables
    return: list with indices where a non-int variable occurs.
    """
    # get all unique j such that dataset[i][j] is not an int, for any j
    indices = {j for j in range(dataset.shape[1])
               for i in range(dataset.shape[0]) if not isint(dataset[i][j])}
    return list(indices)


def get_data(path_to_file=None):
    # open data and separate out variable names, which is the first row
    # TODO - this reads the entire file into memory, which may not be feasible
    # for really large files.
    training_data = open(path_to_file).readlines()
    var_names = training_data[0].split(",")
    training_data = training_data[1:]
    dataset = [item.split(",") for item in training_data]
    training_data = np.array(dataset)
    # extract out answers/labels from the array
    training_data, prices = split_labels_from_dataset(training_data,
                                                      label_idx=-1)

    # get a list of indices to encode, based on which ones aren't integers
    # call an function that encodes these categorical variables
    enc = get_categorical_indices(training_data)
    transformed_data = encode_features(training_data,
                                       indices_to_encode=enc)
    # convert string represetation of int -> int
    nums = np.zeros(transformed_data.shape, dtype=int)
    for i in range(transformed_data.shape[0]):
        nums[i] = np.array(map(float, transformed_data[i]))
    transformed_data = nums
    prices = map(int, prices)
    return var_names, transformed_data, prices

if __name__ == '__main__':
    # paths to datasets
    fpath_descript = "../data/data_description.csv"
    fpath_train = "../data/house_train.csv"
    fpath_test = "../data/house_test.csv"
    var_names, training_data, price_labels = get_data(fpath_train)
    print price_labels[0]
