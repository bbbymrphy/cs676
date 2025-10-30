# search google
import requests
import json

API_KEY = 'AIzaSyABN2rJ--gaFo3cvygAWVGFKwHhmG05upw'
SEARCH_ENGINE_ID = '002c93ab3c3a44f9a' 


search_query = 'back pain'
all_results = []
all_urls = []

for start in range(1, 1001, 10):  # gets up to 100 results (1–100)
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q': search_query,
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'num': 100,       # max per request
        'start': start   # pagination
    }

    response = requests.get(url, params=params)
    results = response.json()

    if "items" in results:
        all_results.extend(results["items"])
        # extract only the 'link' field
        for item in results["items"]:
            all_urls.append(item["link"])

# save full results (optional)
with open('google_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)

# save only the URLs
with open('google_urls.txt', 'w') as f:
    for url in all_urls:
        f.write(url + "\n")

print(f"Collected {len(all_urls)} URLs")
