import joblib
import numpy as np

from features import extract_features

def extract_all_features(url):

    features = extract_features(url)
    features_list=[ 'UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',
            'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',
            'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',
            'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',
            'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick',
            'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain', 'DNSRecording',
            'WebsiteTraffic', 'PageRank', 'GoogleIndex', 'LinksPointingToPage',
            'StatsReport']
    
    main_list=[]
    for check in features_list:
        for feature,items in features.items():
            if check==feature:
                main_list.append(items)
    
    return main_list
    
def model(feature):

    loaded_knn = joblib.load('knn_model.joblib')
    sample_input = np.array([feature])  # Shape: (1, number_of_features)
    prediction = loaded_knn.predict(sample_input)

    print("Predicted Class:", prediction)

    return prediction

if __name__=="__main__":
    url='https://www.youtube.com/watch?v=GswYCadOCaM'
    lis=extract_all_features(url)
    print(len(lis))

