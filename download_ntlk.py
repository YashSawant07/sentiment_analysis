import nltk

nltk.download('stopwords')

import nltk

nltk.download('stopwords', download_dir='nltk_data')

import nltk
import os

nltk_data_dir = 'nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

try:
    from nltk.corpus import stopwords
except LookupError:
    import nltk
    nltk.download('stopwords', download_dir=nltk_data_dir)
    from nltk.corpus import stopwords
