import requests
parameters={"lat": 40.32, "lon":92.12}
response = requests.get("http://api.open-notify.org/iss-pass.json",params=parameters)
print(response.content.decode("utf-8")) #convert content to string from byte representation
print(response.status_code)
    