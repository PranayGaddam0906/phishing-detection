import re
import socket
import requests
import tldextract
import whois
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from googlesearch import search

def boolean_to_int(value):
    """Convert boolean to 1 or -1, return 0 if None."""
    if value is None:
        return 0
    return 1 if value else -1

def check_ip_in_url(url):
    return boolean_to_int(bool(re.match(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url)))

def get_url_length(url):
    return boolean_to_int(len(url) > 75)

def check_shortening_service(url):
    shortening_services = r"(bit\.ly|goo\.gl|tinyurl\.com|ow\.ly|is\.gd|t\.co|t2m\.io)"
    return boolean_to_int(bool(re.search(shortening_services, url)))

def contains_symbol(url, symbol):
    return boolean_to_int(symbol in url)

def count_subdomains(url):
    ext = tldextract.extract(url)
    return boolean_to_int(len(ext.subdomain.split(".")) > 1)

def check_https(url):
    return boolean_to_int(url.startswith("https://"))

def domain_registration_length(domain):
    try:
        domain_info = whois.whois(domain)
        expiration_date = domain_info.expiration_date
        creation_date = domain_info.creation_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        return boolean_to_int((expiration_date - creation_date).days > 365)
    except:
        return 0

def has_favicon(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        return boolean_to_int(bool(soup.find("link", rel="icon") or soup.find("link", rel="shortcut icon")))
    except:
        return 0

def check_non_standard_port(url):
    parsed_url = urlparse(url)
    return boolean_to_int(parsed_url.port not in [80, 443] if parsed_url.port else False)

def check_https_in_domain(url):
    domain = tldextract.extract(url).domain
    return boolean_to_int("https" in domain.lower())

def count_links_in_scripts(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all("script")
        return boolean_to_int(any("http" in script.string for script in scripts if script.string))
    except:
        return 0

def count_links_in_anchors(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        return boolean_to_int(len(soup.find_all("a")) > 20)
    except:
        return 0

def has_popup_window(url):
    try:
        response = requests.get(url, timeout=5)
        return boolean_to_int('window.open' in response.text)
    except:
        return 0

def has_iframe(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        return boolean_to_int(bool(soup.find("iframe")))
    except:
        return 0

def website_traffic(url):
    try:
        return boolean_to_int(len(list(search(url, num_results=10))) > 5)
    except:
        return 0

def google_index(url):
    try:
        query = f"site:{url}"
        return boolean_to_int(len(list(search(query, num_results=1))) > 0)
    except:
        return 0

def dns_record_exists(domain):
    try:
        socket.gethostbyname(domain)
        return boolean_to_int(True)
    except socket.gaierror:
        return boolean_to_int(False)

def abnormal_url(domain, url):
    return boolean_to_int(domain not in url)

def website_forwarding(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=5)
        return boolean_to_int(len(response.history) > 1)
    except:
        return 0

def status_bar_customization(url):
    try:
        response = requests.get(url, timeout=5)
        return boolean_to_int("window.status" in response.text)
    except:
        return 0

def disable_right_click(url):
    try:
        response = requests.get(url, timeout=5)
        return boolean_to_int("event.button==2" in response.text)
    except:
        return 0

def has_info_email(url):
    try:
        response = requests.get(url, timeout=5)
        return boolean_to_int(bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", response.text)))
    except:
        return 0

def count_links_pointing_to_page(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        external_links = [a['href'] for a in soup.find_all("a", href=True) if not a['href'].startswith('#')]
        return boolean_to_int(len(external_links) > 10)
    except:
        return 0

def extract_features(url):
    domain = tldextract.extract(url).registered_domain
    return {
        "UsingIP": check_ip_in_url(url),
        "LongURL": get_url_length(url),
        "ShortURL": check_shortening_service(url),
        "Symbol@": contains_symbol(url, "@"),
        "Redirecting//": contains_symbol(url, "//"),
        "PrefixSuffix-": contains_symbol(url, "-"),
        "SubDomains": count_subdomains(url),
        "HTTPS": check_https(url),
        "DomainRegLen": domain_registration_length(domain),
        "Favicon": has_favicon(url),
        "NonStdPort": check_non_standard_port(url),
        "HTTPSDomainURL": check_https_in_domain(url),
        "RequestURL": count_links_pointing_to_page(url),
        "AnchorURL": count_links_in_anchors(url),
        "LinksInScriptTags": count_links_in_scripts(url),
        "ServerFormHandler": has_popup_window(url),
        "InfoEmail": has_info_email(url),
        "AbnormalURL": abnormal_url(domain, url),
        "WebsiteForwarding": website_forwarding(url),
        "StatusBarCust": status_bar_customization(url),
        "DisableRightClick": disable_right_click(url),
        "UsingPopupWindow": has_popup_window(url),
        "IframeRedirection": has_iframe(url),
        "AgeofDomain": domain_registration_length(domain),
        "DNSRecording": dns_record_exists(domain),
        "WebsiteTraffic": website_traffic(url),
        "PageRank": website_traffic(url),
        "GoogleIndex": google_index(url),
        "LinksPointingToPage": count_links_pointing_to_page(url),
        "StatsReport": google_index(url)
    }


