import json
THROW_AWAY = ['yearAge', 'filteredAddress',
		'countyName', 'stateOrProvinceName', 'photoURL',
		 'MLSNumber', 'listingStatus', 'publicRemarks',
		 'siteMapDetailUrlPath', 'yearBuilt']

#data extraction code
#this file won't run because "sj.json" wasn't pushed.
#it contained information such as the address and comments which
#weren't needed for this project. This code just parses that stuff out
with open('sj.json') as data_file:
	data = json.load(data_file)

li = data['propertySearchResults']

for x in xrange(0, len(li)):
	item = li[x]
	li[x] = {k: v for k,v in item.iteritems() if k not in THROW_AWAY}



with open('sjout.json', 'w') as outfile:
	json.dump(data, outfile)

#test
with open('sjout.json', 'r') as newf:
	d = json.load(newf)
li = d['propertySearchResults']
print li[0]
