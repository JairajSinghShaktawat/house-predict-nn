import json
import requests
parameters={"lat": 40.32, "lon":92.12}
response = requests.get("http://api.open-notify.org/astros.json",params=parameters)
#print the response data
#print(response.content.decode("utf-8")) #convert content to string from byte representation
#print(response.status_code)

#get the JSON data
data=response.json()
print response.headers
print response.headers["content-type"]
print data["number"],
print "currently in space"

def recursive_api_call(base_url,param_names, param_args):
	"""base_url is the URL that our api request will go to
	names are the names of the parameters while param_args are the arguments. 
	"""
	parameters=dict(zip(param_names,param_args))
	response=response.get(base_url,parameters)
	if(response.status_code==200):
		data=response.json()
		response_dict=json.loads(data)
		#do stuff with the response dictionary
		#this method can be called recursively

