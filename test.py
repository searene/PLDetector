import requests
from requests.auth import HTTPBasicAuth

from src import config

r = requests.get("https://api.github.com/user/repos", auth=HTTPBasicAuth(config.github_username, config.github_password))
print(r.text)
