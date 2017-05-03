import pyhs2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk.corpus
import numpy as np
import numpy.linalg as LA
import requests
import json
import feedparser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from textwrap import wrap

np.seterr(divide='ignore', invalid='ignore')


def tokenize_and_crawl(article):
    """
    Function called if the news article not found in the hive database, crawl the web for it.
    :param article:
    :return: None
    """

    # Tokenize and lemmatize the input article
    article_string = ""

    for i in article:
        article_string += i
        article_string += " "

    tokens = word_tokenize(article_string)

    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Stemming to a common root

    stop_w = stopwords.words('english')
    w = []

    # Remove stop words
    for z in tokens:
        if z not in stop_w:
            w.append(z)

    query_string = ""

    for o in w:
        query_string += o
        query_string += "+"

    # Crawl using google news API
    print "Keywords used to crawl NEWS sites for this article are :", query_string
    crawl_url = "http://news.google.com/news?q=" + query_string + "&output=rss"
    feed = feedparser.parse(crawl_url)

    corpus = []

    print "------------------------------------------------------"
    print "CRAWLING RESULTS : "
    for x in feed.entries:
        print x['title']
        corpus.append(x['title'])

    string_cosine_correlation_level3(article, corpus)


def string_cosine_correlation_level3(article, corpus):
    """
    Defines the correlation between the article and a corpus.
    Method calculates the cosine distance between each term in the article and each term in the corpus.
    It will generate a likely hood match between strings based on their cosine distance.
    Libraries used : Numpy, Scikit-learn, nltk
    :param article:
    :param corpus:
    :return: None
    """

    plot_dict = {}

    stopWords = stopwords.words('english')

    vectorize = CountVectorizer(stop_words=stopWords)

    transformer = TfidfTransformer()

    corpus_array = vectorize.fit_transform(corpus).toarray()
    article_array = vectorize.transform(article).toarray()

    """
    Lambda function to calculate the cosine distance between two vectors.
    Inner product(x, y) = arccosine(x * y / |x|*|y|)
    np.inner(a, b) : Gives the dot product (x * y).
    (LA.norm(a) * LA.norm(b) : Gives |x|*|y|
    """
    cx = lambda a, b: round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)

    print "LEVEL-3 SEARCH : WEB CRAWLING"
    z = 0
    for vector in corpus_array:
        for testV in article_array:
            cosine = cx(vector, testV)

            if cosine > 0:
                print "Cosine match value : ", cosine, ",", "Corpus Index is :", z
                print "Original article is :", corpus[z]
                plot_dict[corpus[z]] = cosine

        z += 1

    """
    Generate 2-D plot which plots the article against the cosine distance
    """
    generate_plot_from_dictionary(plot_dict, "News feed crawling cosine comparision")

    transformer.fit(corpus_array)
    transformer.transform(corpus_array).toarray()
    transformer.fit(article_array)
    tfidf = transformer.transform(article_array)
    tfidf.todense()


def string_cosine_correlation_level2(article, corpus):
    """
    Defines the correlation between the article and a corpus.
    Method calculates the cosine distance between each term in the article and each term in the corpus.
    It will generate a likely hood match between strings based on their cosine distance.
    Libraries used : Numpy, Scikit-learn, nltk
    :param article:
    :param corpus:
    :return: None
    """

    plot_dict = {}

    stopWords = nltk.corpus.stopwords.words('english')

    """
    Convert a collection of text documents to a matrix of token counts.
    Parameter passed are english stopwords. If any token match the stop words, its removed.
    """
    vectorize = CountVectorizer(stop_words=stopWords)

    transformer = TfidfTransformer()

    corpus_array = vectorize.fit_transform(corpus).toarray()  # IDF
    article_array = vectorize.transform(article).toarray()  # TF

    cx = lambda a, b: round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)

    print "LEVEL-2 SEARCH : HIVE METASTORE"
    z = 0
    for vector in corpus_array:
        for testV in article_array:
            cosine = cx(vector, testV)

            if cosine > 0:
                print "Cosine match value : ", cosine, ",", "Corpus Index is :", z
                print "Original article is :", corpus[z]
                plot_dict[corpus[z]] = cosine

        z += 1

    """
    Generate 2-D plot which plots the article against the cosine distance
    """
    generate_plot_from_dictionary(plot_dict, "Hive Metastore comparision plot")

    tokenize_and_crawl(article)

    transformer.fit(corpus_array)
    transformer.transform(corpus_array).toarray()
    transformer.fit(article_array)
    tfidf = transformer.transform(article_array)
    tfidf.todense()


def generate_plot_from_dictionary(plot_dict, plot_title):
    plt.xlabel("Wall posts")
    plt.ylabel("Reliability fraction (Cosine distance)")
    plt.title(plot_title)

    x_ticks = ['\n'.join(wrap(l, 20)) for l in plot_dict.keys()]
    plt.bar(range(len(plot_dict)), plot_dict.values(), align='center')
    plt.xticks(range(len(plot_dict)), x_ticks)

    # Show the plot
    plt.show()


def get_facebook_data():
    """
    Function calls the facebook GraphAPI using curl URL.
    A JSON is returned with the required fields.
    This JSON is converted into an array(list) and returned
    :return: List
    """
    temp = []
    news_keywords = ['battery', 'Police', 'strong', 'Justice', 'Egypt', 'army', 'empire', 'IPL']

    r = requests.get(
        "https://graph.facebook.com/v2.9/me?fields=id%2Cname%2Cposts%7Bmessage%2Cdescription%2Csource%2Cid%7D&access_token=EAACEdEose0cBAIesXmkMSitLER2CNQvjIMAISzMCOZAVsl26Ba5ZAGA9pyCNeLAIPf2qxcuIgsiJHwzcFuGSUmdEueVrCdUqyquylm5plbDLcGSFefx5Pxjax2uc9ZCAoJcUyjLahUpMbtWlKbf7PUeYqVGeAlRUJaU1rDH2U3CuWfmAmpkCqoqcUt2sycZD")

    data = json.loads(r.text)

    for z in data['posts']['data']:
        if 'description' in z.keys():
            temp.append(z['description'])

    news = []

    for x in temp:
        for y in news_keywords:
            if y in x:
                news.append(str(x))

    article = []
    for x in set(news):
        article.append(x)

    y = [article[1]]
    return y


def hive_news_dump_connection():
    with pyhs2.connect(host='192.168.56.101',
                       port=10000,
                       authMechanism='PLAIN',
                       user='hiveuser',
                       password='password',
                       database='anuvrat') as conn:

        temp_query = 'SELECT name, message, description FROM ABC_NEWS LIMIT 10'
        with conn.cursor() as cur:
            cur.execute(temp_query)

            corpus = []

            for x in cur.fetchall():
                for i in x:
                    i = i.replace('\xef\xbf\xbd\xef\xbf\xbd', '')
                    corpus.append(i.replace('\x00', ''))

            return corpus


def main():
    # Call the function to get article from Facebook.
    article = get_facebook_data()
    # Call function to connect to hive metastore and get news dump.
    corpus = hive_news_dump_connection()

    print "Article : ", article
    print "--------------------------------------------------------"
    print corpus

    string_cosine_correlation_level2(article, corpus)


if __name__ == '__main__':
    main()
