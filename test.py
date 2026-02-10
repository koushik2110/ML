import requests
from bs4 import BeautifulSoup

url = "https://docs.oracle.com/en/cloud/saas/readiness/hcm/25b/payr-25b/index.html"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

print(soup)
