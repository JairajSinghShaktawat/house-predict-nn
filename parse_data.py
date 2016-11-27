import json


class DataParser(object):
    def __init__(self, json_file, THROW_AWAY=[]):
        with open(json_file, "r") as data_file:
            data = json.load(data_file)
        filtered_data = data["propertySearchResults"]
        for x in range(0, len(filtered_data)):
            item = filtered_data[x]
            filtered_data[x] = {k: v for k, v in item.iteritems() if k not
                                in THROW_AWAY}
        self.filtered_data = filtered_data

    def get_data_as_dict(self):
        return self.filtered_data

if __name__ == '__main__':
    THROW_AWAY = ['yearAge', 'filteredAddress', 'countyName', 'stateOrProvinceName', 'photoURL', 'MLSNumber', 'listingStatus', 'publicRemarks', 'siteMapDetailUrlPath', 'yearBuilt']
    parser = DataParser("sj.json", THROW_AWAY)
    print parser.filtered_data
