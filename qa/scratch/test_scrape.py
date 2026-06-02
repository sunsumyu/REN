import urllib.request
import urllib.parse

def test_scrape(drug_name):
    # 39 health network search url
    search_url = f"https://yp.39.net/search/{urllib.parse.quote(drug_name)}.shtml"
    req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=5.0) as r:
            html = r.read().decode('utf-8', errors='ignore')
            print(f"Success for {drug_name}, HTML length: {len(html)}")
            if "愈肝片" in html:
                print("Found drug name in HTML!")
            return True
    except Exception as e:
        print(f"Error fetching {search_url}: {e}")
        return False

test_scrape("愈肝片")
test_scrape("獾油")
