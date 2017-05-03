# code_fake_news
Giving a reliability index of a new item shared on a users wall.
1. A dump of 40000 records for news dumps are present in Hive database.
2. User data is pulled from Facebook graph API.
3. Corpus is generated from the Hive database and article is pulled from GraphAPI.
4. Term frquency is generated from article.
5. Inverse document frequency generated from corpus.
6. Cosine distance is calculted between each item in article and corpus.
7. If cosine distance is close to 1, article matches a corpus entry.
8. Matplotlib is used to visualize the results.
9. Tokenization, lemmatization, porter stemming is done on the article keywords.
10. If article is not found in Hive, google news API is crawled for lemmatized keywords.
11. Results from thr RSS feed are used to build the new corpus.
12. Repeat step 4 to 7 and visualize the results.

Summary :
Tokenization : Use of Scikit-learn to get keywords and eliminate stopwords.
Lemmatization : Using stemming to find root lemma of words.
Inner product : Cosine distance between article and corpus generated using inner product.
Term frequency : TF for article matrix.
Inverse Document frequency : IDF for article matrix.
Matplotlib : To visualize cosine distance.
Hive : Metastore for news dump used in level-2 verification.
Most of the above tools can be found in Python NLTK (Natural language processing toolkit) and Sklearn (Scikit-learn).


## Dataset link : https://www.dropbox.com/s/vdq2ksi73jhwh22/11_new_agencies-facebook-posts.zip?dl=0
