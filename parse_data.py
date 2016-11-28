import json
import argparse


class DataParser(object):

    def __init__(self, json_file, THROW_AWAY=set()):
        with open(json_file, "r") as data_file:
            data = json.load(data_file)
        filtered_data = data["propertySearchResults"]
        for x in range(0, len(filtered_data)):
            item = filtered_data[x]
            filtered_data[x] = {k: v for k, v in item.iteritems() if k not
                                in THROW_AWAY}
        self.filtered_data = filtered_data

    def get_data(self):
        return self.filtered_data

    def dict_to_matrix(self, list):
        li = []
        for dict_data in list:
            features = [v for k, v in dict_data.iteritems() if
                        k != "listSalePrice"]

            features.append(dict_data["listSalePrice"])
            li.append(features)
        return li


# def remove_throw_out(throw_out): return lambda (k, v): (k, v) if k not in
# throw_out
#
#
# def get_data(json_file, throw_out=set()):
#     with open(json_file, "r") as data_file:
#         data = json.load(data_file)
#     filtered_data = data["propertySearchResults"]
#     filtered_data = map(remove_throw_out, filtered_data)
#     return filtered_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("f", type=str)
    args = parser.parse_args()
    THROW_AWAY = ('yearAge', 'filteredAddress', 'countyName',
                  'stateOrProvinceName', 'photoURL', 'MLSNumber',
                  'listingStatus', 'publicRemarks', 'siteMapDetailUrlPath',
                  'yearBuilt')
    THROW_AWAY = set(THROW_AWAY)
    data_parser = DataParser(args.f, THROW_AWAY)
    # print parser.filtered_data[0]
    features_list = data_parser.dict_to_matrix(data_parser.get_data())
    print len(features_list)
