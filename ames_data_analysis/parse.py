from sklearn import preprocessing
import numpy as np




if __name__ == '__main__':
    fpath_descript = "../data/data_description.csv"
    fpath_train = "../data/house_train.csv"
    fpath_test = "../data/house_test.csv"
    training_data = []
    with open(fpath_train) as file:
        training_data = file.readlines()
    var_names = training_data[0]
    var_names = var_names.split(",")
    print var_names
    dataset = []
    for item in training_data:
        dataset.append(item.split(","))
    training_data = dataset

    enc = preprocessing.OneHotEncoder()
