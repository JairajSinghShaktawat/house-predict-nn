import json

#data extraction code
#this file won't run because "sj.json" wasn't pushed.
#it contained information such as the address and comments which 
#weren't needed for this project. This code just parses that stuff out 
with open('sj.json') as data_file:
	data = json.load(data_file)
li = data['propertySearchResults']
for item in li:
	keys = item.keys()
	for k in keys:
		if k in ['yearAge', 'filteredAddress', 
		'countyName', 'stateOrProvinceName', 'photoURL',
		 'MLSNumber', 'listingStatus', 'publicRemarks', 
		 'siteMapDetailUrlPath', 'yearBuilt']:
		 	item.pop(k)



with open('sjout.json', 'w') as outfile:
	json.dump(data, outfile)

#test
with open('sjout.json', 'r') as newf:
	d = json.load(newf)
li = d['propertySearchResults']
print li[0]
	