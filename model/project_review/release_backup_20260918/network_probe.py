import concurrent.futures
import urllib.request


def probe(url):
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'MRMPFormer-release'})
        with urllib.request.urlopen(request, timeout=12) as response:
            print(url.split('/')[2], response.status, response.headers.get('Content-Type'))
    except Exception as error:
        print(type(error).__name__, str(error))


with concurrent.futures.ThreadPoolExecutor(2) as pool:
    list(pool.map(probe, [
        'https://api.github.com/repos/gamstek/MRMPFormer',
        'https://github.com/gamstek/MRMPFormer.git/info/refs?service=git-upload-pack',
    ]))
