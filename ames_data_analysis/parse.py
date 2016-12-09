




if __name__ == '__main__':
    fpath_descript = "../data/data_description.csv"
    fpath_train = "../data/house_train.csv"
    fpath_test = "../data/house_test.csv"
    training_data = []
    with open(fpath_train) as file:
        training_data = file.readlines()
    var_names = training_data[0]
    training_data = training_data[1:]
    print training_data[0]
