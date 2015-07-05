
# coding: utf-8

# version 1.0.3
# #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# # **Text Analysis and Entity Resolution**
# ####Entity resolution is a common, yet difficult problem in data cleaning and integration. This lab will demonstrate how we can use Apache Spark to apply powerful and scalable text analysis techniques and perform entity resolution across two datasets of commercial products.

# #### Entity Resolution, or "[Record linkage][wiki]" is the term used by statisticians, epidemiologists, and historians, among others, to describe the process of joining records from one data source with another that describe the same entity. Our terms with the same meaning include, "entity disambiguation/linking", duplicate detection", "deduplication", "record matching", "(reference) reconciliation", "object identification", "data/information integration", and "conflation".
# #### Entity Resolution (ER) refers to the task of finding records in a dataset that refer to the same entity across different data sources (e.g., data files, books, websites, databases). ER is necessary when joining datasets based on entities that may or may not share a common identifier (e.g., database key, URI, National identification number), as may be the case due to differences in record shape, storage location, and/or curator style or preference. A dataset that has undergone ER may be referred to as being cross-linked.
# [wiki]: https://en.wikipedia.org/wiki/Record_linkage

# ### Code
# #### This assignment can be completed using basic Python, pySpark Transformations and actions, and the plotting library matplotlib. Other libraries are not allowed.
# ### Files
# #### Data files for this assignment are from the [metric-learning](https://code.google.com/p/metric-learning/) project and can be found at:
# `cs100/lab3`
# #### The directory contains the following files:
# * **Google.csv**, the Google Products dataset
# * **Amazon.csv**, the Amazon dataset
# * **Google_small.csv**, 200 records sampled from the Google data
# * **Amazon_small.csv**, 200 records sampled from the Amazon data
# * **Amazon_Google_perfectMapping.csv**, the "gold standard" mapping
# * **stopwords.txt**, a list of common English words
# #### Besides the complete data files, there are "sample" data files for each dataset - we will use these for **Part 1**. In addition, there is a "gold standard" file that contains all of the true mappings between entities in the two datasets. Every row in the gold standard file has a pair of record IDs (one Google, one Amazon) that belong to two record that describe the same thing in the real world. We will use the gold standard to evaluate our algorithms.

# ### **Part 0: Preliminaries**
# #### We read in each of the files and create an RDD consisting of lines.
# #### For each of the data files ("Google.csv", "Amazon.csv", and the samples), we want to parse the IDs out of each record. The IDs are the first column of the file (they are URLs for Google, and alphanumeric strings for Amazon). Omitting the headers, we load these data files into pair RDDs where the *mapping ID* is the key, and the value is a string consisting of the name/title, description, and manufacturer from the record.
# #### The file format of an Amazon line is:
#    `"id","title","description","manufacturer","price"`
# #### The file format of a Google line is:
#    `"id","name","description","manufacturer","price"`

# In[1]:

import re
DATAFILE_PATTERN = '^(.+),"(.+)",(.*),(.*),(.*)'

def removeQuotes(s):
    """ Remove quotation marks from an input string
    Args:
        s (str): input string that might have the quote "" characters
    Returns:
        str: a string without the quote characters
    """
    return ''.join(i for i in s if i!='"')


def parseDatafileLine(datafileLine):
    """ Parse a line of the data file using the specified regular expression pattern
    Args:
        datafileLine (str): input string that is a line from the data file
    Returns:
        str: a string parsed using the given regular expression and without the quote characters
    """
    match = re.search(DATAFILE_PATTERN, datafileLine)
    if match is None:
        print 'Invalid datafile line: %s' % datafileLine
        return (datafileLine, -1)
    elif match.group(1) == '"id"':
        print 'Header datafile line: %s' % datafileLine
        return (datafileLine, 0)
    else:
        product = '%s %s %s' % (match.group(2), match.group(3), match.group(4))
        return ((removeQuotes(match.group(1)), product), 1)


# In[2]:

import sys
import os
from test_helper import Test

baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab3')

GOOGLE_PATH = 'Google.csv'
GOOGLE_SMALL_PATH = 'Google_small.csv'
AMAZON_PATH = 'Amazon.csv'
AMAZON_SMALL_PATH = 'Amazon_small.csv'
GOLD_STANDARD_PATH = 'Amazon_Google_perfectMapping.csv'
STOPWORDS_PATH = 'stopwords.txt'

def parseData(filename):
    """ Parse a data file
    Args:
        filename (str): input file name of the data file
    Returns:
        RDD: a RDD of parsed lines
    """
    return (sc
            .textFile(filename, 4, 0)
            .map(parseDatafileLine)
            .cache())

def loadData(path):
    """ Load a data file
    Args:
        path (str): input file name of the data file
    Returns:
        RDD: a RDD of parsed valid lines
    """
    filename = os.path.join(baseDir, inputPath, path)
    raw = parseData(filename).cache()
    failed = (raw
              .filter(lambda s: s[1] == -1)
              .map(lambda s: s[0]))
    for line in failed.take(10):
        print '%s - Invalid datafile line: %s' % (path, line)
    valid = (raw
             .filter(lambda s: s[1] == 1)
             .map(lambda s: s[0])
             .cache())
    print '%s - Read %d lines, successfully parsed %d lines, failed to parse %d lines' % (path,
                                                                                        raw.count(),
                                                                                        valid.count(),
                                                                                        failed.count())
    assert failed.count() == 0
    assert raw.count() == (valid.count() + 1)
    return valid

googleSmall = loadData(GOOGLE_SMALL_PATH)
google = loadData(GOOGLE_PATH)
amazonSmall = loadData(AMAZON_SMALL_PATH)
amazon = loadData(AMAZON_PATH)


# #### Let's examine the lines that were just loaded in the two subset (small) files - one from Google and one from Amazon

# In[3]:

for line in googleSmall.take(3):
    print 'google: %s: %s\n' % (line[0], line[1])

for line in amazonSmall.take(3):
    print 'amazon: %s: %s\n' % (line[0], line[1])


# ### **Part 1: ER as Text Similarity - Bags of Words**
# #### A simple approach to entity resolution is to treat all records as strings and compute their similarity with a string distance function. In this part, we will build some components for performing bag-of-words text-analysis, and then use them to compute record similarity.
# #### [Bag-of-words][bag-of-words] is a conceptually simple yet powerful approach to text analysis.
# #### The idea is to treat strings, a.k.a. **documents**, as *unordered collections* of words, or **tokens**, i.e., as bags of words.
# > #### **Note on terminology**: a "token" is the result of parsing the document down to the elements we consider "atomic" for the task at hand.  Tokens can be things like words, numbers, acronyms, or other exotica like word-roots or fixed-length character strings.
# > #### Bag of words techniques all apply to any sort of token, so when we say "bag-of-words" we really mean "bag-of-tokens," strictly speaking.
# #### Tokens become the atomic unit of text comparison. If we want to compare two documents, we count how many tokens they share in common. If we want to search for documents with keyword queries (this is what Google does), then we turn the keywords into tokens and find documents that contain them. The power of this approach is that it makes string comparisons insensitive to small differences that probably do not affect meaning much, for example, punctuation and word order.
# [bag-of-words]: https://en.wikipedia.org/wiki/Bag-of-words_model

# ### **1(a) Tokenize a String**
# #### Implement the function `simpleTokenize(string)` that takes a string and returns a list of non-empty tokens in the string. `simpleTokenize` should split strings using the provided regular expression. Since we want to make token-matching case insensitive, make sure all tokens are turned lower-case. Give an interpretation, in natural language, of what the regular expression, `split_regex`, matches.
# #### If you need help with Regular Expressions, try the site [regex101](https://regex101.com/) where you can interactively explore the results of applying different regular expressions to strings. *Note that \W includes the "_" character*.  You should use [re.split()](https://docs.python.org/2/library/re.html#re.split) to perform the string split. Also, make sure you remove any empty tokens.

# In[4]:

# TODO: Replace <FILL IN> with appropriate code
import re
quickbrownfox = 'A quick brown fox jumps over the lazy dog.'
split_regex = r'\W+'

def simpleTokenize(string):
    """ A simple implementation of input string tokenization
    Args:
        string (str): input string
    Returns:
        list: a list of tokens
    """
    string = string.lower()   
        
    splt = re.split(split_regex, string)
    
    fltr = filter(None,splt)
    
    return fltr

#print simpleTokenize(quickbrownfox)
print simpleTokenize(quickbrownfox) # Should give ['a', 'quick', 'brown', ... ]


# In[5]:

# TEST Tokenize a String (1a)
Test.assertEquals(simpleTokenize(quickbrownfox),
                  ['a','quick','brown','fox','jumps','over','the','lazy','dog'],
                  'simpleTokenize should handle sample text')
Test.assertEquals(simpleTokenize(' '), [], 'simpleTokenize should handle empty string')
Test.assertEquals(simpleTokenize('!!!!123A/456_B/789C.123A'), ['123a','456_b','789c','123a'],
                  'simpleTokenize should handle puntuations and lowercase result')
Test.assertEquals(simpleTokenize('fox fox'), ['fox', 'fox'],
                  'simpleTokenize should not remove duplicates')


# ### **(1b) Removing stopwords**
# #### *[Stopwords][stopwords]* are common (English) words that do not contribute much to the content or meaning of a document (e.g., "the", "a", "is", "to", etc.). Stopwords add noise to bag-of-words comparisons, so they are usually excluded.
# #### Using the included file "stopwords.txt", implement `tokenize`, an improved tokenizer that does not emit stopwords.
# [stopwords]: https://en.wikipedia.org/wiki/Stop_words

# In[6]:

# TODO: Replace <FILL IN> with appropriate code
stopfile = os.path.join(baseDir, inputPath, STOPWORDS_PATH)
stopwords = set(sc.textFile(stopfile).collect())
#print 'These are the stopwords: %s' % stopwords

def tokenize(string):
    """ An implementation of input string tokenization that excludes stopwords
    Args:
        string (str): input string
    Returns:
        list: a list of tokens without stopwords
    """
    string = string.lower()   
        
    splt = re.split(split_regex, string)
    
    fltr = filter(lambda x: x not in stopwords,splt)
    
    fltr = filter(None,fltr)
       
    return fltr

print tokenize(quickbrownfox) # Should give ['quick', 'brown', ... ]


# In[7]:

# TEST Removing stopwords (1b)
Test.assertEquals(tokenize("Why a the?"), [], 'tokenize should remove all stopwords')
Test.assertEquals(tokenize("Being at the_?"), ['the_'], 'tokenize should handle non-stopwords')
Test.assertEquals(tokenize(quickbrownfox), ['quick','brown','fox','jumps','lazy','dog'],
                    'tokenize should handle sample text')


# ### **(1c) Tokenizing the small datasets**
# #### Now let's tokenize the two *small* datasets. For each ID in a dataset, `tokenize` the values, and then count the total number of tokens.
# #### How many tokens, total, are there in the two datasets?

# In[8]:

# TODO: Replace <FILL IN> with appropriate code
amazonRecToToken = amazonSmall.map(lambda x: (x[0],tokenize(x[1])))
googleRecToToken = googleSmall.map(lambda x: (x[0],tokenize(x[1])))

def countTokens(vendorRDD):
    """ Count and return the number of tokens
    Args:
        vendorRDD (RDD of (recordId, tokenizedValue)): Pair tuple of record ID to tokenized output
    Returns:
        count: count of all tokens
    """
    
    #token = max(vendorRDD.map(lambda x: len(x)).collect())
    
    return vendorRDD.mapValues(lambda s: len(s)).values().sum()
    
totalTokens = countTokens(amazonRecToToken) + countTokens(googleRecToToken)
print 'There are %s tokens in the combined datasets' % totalTokens


# In[9]:

# TEST Tokenizing the small datasets (1c)
Test.assertEquals(totalTokens, 22520, 'incorrect totalTokens')


# ### **(1d) Amazon record with the most tokens**
# #### Which Amazon record has the biggest number of tokens?
# #### In other words, you want to sort the records and get the one with the largest count of tokens.

# In[10]:

# TODO: Replace <FILL IN> with appropriate code
def findBiggestRecord(vendorRDD):
    """ Find and return the record with the largest number of tokens
    Args:
        vendorRDD (RDD of (recordId, tokens)): input Pair Tuple of record ID and tokens
    Returns:
        list: a list of 1 Pair Tuple of record ID and tokens
    """
    val = vendorRDD.map(lambda (a,b): (len(b), (a,b))).sortByKey(False)
       
    val1 = val.map(lambda (a,b): b).map(lambda(a,b):[a,b]).take(1) 
                     
    return val1

biggestRecordAmazon = findBiggestRecord(amazonRecToToken)
#print biggestRecordAmazon

print 'The Amazon record with ID "%s" has the most tokens (%s)' % (biggestRecordAmazon[0][0],
                                                                   len(biggestRecordAmazon[0][1]))


# In[11]:

# TEST Amazon record with the most tokens (1d)
Test.assertEquals(biggestRecordAmazon[0][0], 'b000o24l3q', 'incorrect biggestRecordAmazon')
Test.assertEquals(len(biggestRecordAmazon[0][1]), 1547, 'incorrect len for biggestRecordAmazon')


# ### **Part 2: ER as Text Similarity - Weighted Bag-of-Words using TF-IDF**
# #### Bag-of-words comparisons are not very good when all tokens are treated the same: some tokens are more important than others. Weights give us a way to specify which tokens to favor. With weights, when we compare documents, instead of counting common tokens, we sum up the weights of common tokens. A good heuristic for assigning weights is called "Term-Frequency/Inverse-Document-Frequency," or [TF-IDF][tfidf] for short.
# #### **TF**
# #### TF rewards tokens that appear many times in the same document. It is computed as the frequency of a token in a document, that is, if document *d* contains 100 tokens and token *t* appears in *d* 5 times, then the TF weight of *t* in *d* is *5/100 = 1/20*. The intuition for TF is that if a word occurs often in a document, then it is more important to the meaning of the document.
# #### **IDF**
# #### IDF rewards tokens that are rare overall in a dataset. The intuition is that it is more significant if two documents share a rare word than a common one. IDF weight for a token, *t*, in a set of documents, *U*, is computed as follows:
# * #### Let *N* be the total number of documents in *U*
# * #### Find *n(t)*, the number of documents in *U* that contain *t*
# * #### Then *IDF(t) = N/n(t)*.
# #### Note that *n(t)/N* is the frequency of *t* in *U*, and *N/n(t)* is the inverse frequency.
# > #### **Note on terminology**: Sometimes token weights depend on the document the token belongs to, that is, the same token may have a different weight when it's found in different documents.  We call these weights *local* weights.  TF is an example of a local weight, because it depends on the length of the source.  On the other hand, some token weights only depend on the token, and are the same everywhere that token is found.  We call these weights *global*, and IDF is one such weight.
# #### **TF-IDF**
# #### Finally, to bring it all together, the total TF-IDF weight for a token in a document is the product of its TF and IDF weights.
# [tfidf]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

# ### **(2a) Implement a TF function**
# #### Implement `tf(tokens)` that takes a list of tokens and returns a Python [dictionary](https://docs.python.org/2/tutorial/datastructures.html#dictionaries) mapping tokens to TF weights.
# #### The steps your function should perform are:
# * #### Create an empty Python dictionary
# * #### For each of the tokens in the input `tokens` list, count 1 for each occurance and add the token to the dictionary
# * #### For each of the tokens in the dictionary, divide the token's count by the total number of tokens in the input `tokens` list

# In[45]:

# TODO: Replace <FILL IN> with appropriate code
def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    Dict = {}
    j = float(len(tokens))
    return dict((i,tokens.count(i)/j) for i in tokens)

print tf(tokenize(quickbrownfox)) # Should give { 'quick': 0.1666 ... }


# In[46]:

# TEST Implement a TF function (2a)
tf_test = tf(tokenize(quickbrownfox))
Test.assertEquals(tf_test, {'brown': 0.16666666666666666, 'lazy': 0.16666666666666666,
                             'jumps': 0.16666666666666666, 'fox': 0.16666666666666666,
                             'dog': 0.16666666666666666, 'quick': 0.16666666666666666},
                    'incorrect result for tf on sample text')
tf_test2 = tf(tokenize('one_ one_ two!'))
Test.assertEquals(tf_test2, {'one_': 0.6666666666666666, 'two': 0.3333333333333333},
                    'incorrect result for tf test')


# ### **(2b) Create a corpus**
# #### Create a pair RDD called `corpusRDD`, consisting of a combination of the two small datasets, `amazonRecToToken` and `googleRecToToken`. Each element of the `corpusRDD` should be a pair consisting of a key from one of the small datasets (ID or URL) and the value is the associated value for that key from the small datasets.

# In[14]:

# TODO: Replace <FILL IN> with appropriate code
corpusRDD = (amazonRecToToken.values().union(googleRecToToken.values()))

uniqueTokens = corpusRDD.flatMap(lambda x:list(set(x[1])))
print corpusRDD.take(2)
print uniqueTokens.take(40)

# TEST Create a corpus (2b)
Test.assertEquals(corpusRDD.count(), 400, 'incorrect corpusRDD.count()')
# ### **(2c) Implement an IDFs function**
# #### Implement `idfs` that assigns an IDF weight to every unique token in an RDD called `corpus`. The function should return an pair RDD where the `key` is the unique token and value is the IDF weight for the token.
# #### Recall that the IDF weight for a token, *t*, in a set of documents, *U*, is computed as follows:
# * #### Let *N* be the total number of documents in *U*.
# * #### Find *n(t)*, the number of documents in *U* that contain *t*.
# * #### Then *IDF(t) = N/n(t)*.
# #### The steps your function should perform are:
# * #### Calculate *N*. Think about how you can calculate *N* from the input RDD.
# * #### Create an RDD (*not a pair RDD*) containing the unique tokens from each document in the input `corpus`. For each document, you should only include a token once, *even if it appears multiple times in that document.*
# * #### For each of the unique tokens, count how many times it appears in the document and then compute the IDF for that token: *N/n(t)*
# #### Use your `idfs` to compute the IDF weights for all tokens in `corpusRDD` (the combined small datasets).
# #### How many unique tokens are there?

# In[15]:

# TODO: Replace <FILL IN> with appropriate code
def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """
    #N = <FILL IN>
    #uniqueTokens = corpus.<FILL IN>
    
    N = float(corpus.count())
    uniqueTokens = corpus.flatMap(lambda x:list(set(x[1])))
    tokenCountPairTuple = uniqueTokens.map(lambda a: (a, 1))
    tokenSumPairTuple = tokenCountPairTuple.reduceByKey(lambda a,b: a + b)
    return tokenSumPairTuple.map(lambda (a,b): (a,N/b))
        
    #tokenCountPairTuple = uniqueTokens.<FILL IN>
    #tokenSumPairTuple = tokenCountPairTuple.<FILL IN>
    #return (tokenSumPairTuple.<FILL IN>)
 
idfsSmall = idfs(amazonRecToToken.union(googleRecToToken))
uniqueTokenCount = idfsSmall.count()

print idfsSmall.takeOrdered(1, lambda s: s[1])[0]

print 'There are %s unique tokens in the small datasets.' % uniqueTokenCount


# In[16]:

# TEST Implement an IDFs function (2c)
Test.assertEquals(uniqueTokenCount, 4772, 'incorrect uniqueTokenCount')
tokenSmallestIdf = idfsSmall.takeOrdered(1, lambda s: s[1])[0]
Test.assertEquals(tokenSmallestIdf[0], 'software', 'incorrect smallest IDF token')
Test.assertTrue(abs(tokenSmallestIdf[1] - 4.25531914894) < 0.0000000001,
                'incorrect smallest IDF value')


# ### **(2d) Tokens with the smallest IDF**
# #### Print out the 11 tokens with the smallest IDF in the combined small dataset.

# In[17]:

smallIDFTokens = idfsSmall.takeOrdered(11, lambda s: s[1])
print smallIDFTokens


# ### **(2e) IDF Histogram**
# #### Plot a histogram of IDF values.  Be sure to use appropriate scaling and bucketing for the data.
# #### First plot the histogram using `matplotlib`

# In[18]:

import matplotlib.pyplot as plt

small_idf_values = idfsSmall.map(lambda s: s[1]).collect()
fig = plt.figure(figsize=(8,3))
plt.hist(small_idf_values, 50, log=True)
pass


# ### **(2f) Implement a TF-IDF function**
# #### Use your `tf` function to implement a `tfidf(tokens, idfs)` function that takes a list of tokens from a document and a Python dictionary of IDF weights and returns a Python dictionary mapping individual tokens to total TF-IDF weights.
# #### The steps your function should perform are:
# * #### Calculate the token frequencies (TF) for `tokens`
# * #### Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight
# #### Use your `tfidf` function to compute the weights of Amazon product record 'b000hkgj8k'. To do this, we need to extract the record for the token from the tokenized small Amazon dataset and we need to convert the IDFs for the small dataset into a Python dictionary. We can do the first part, by using a `filter()` transformation to extract the matching record and a `collect()` action to return the value to the driver. For the second part, we use the [`collectAsMap()` action](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.collectAsMap) to return the IDFs to the driver as a Python dictionary.

# In[67]:

# TODO: Replace <FILL IN> with appropriate code
def tfidf(tokens, idfs):
    """ Compute TF-IDF
    Args:
        tokens (list of str): input list of tokens from tokenize
        idfs (dictionary): record to IDF value
    Returns:
        dictionary: a dictionary of records to TF-IDF values
    """
    tfs = tf(tokens) #'autocad': 0.16666666666666666,
    tfIdfDict = {}
    for tfs_key, tfs_value in tfs.iteritems():
        for idfs_key, idfs_value in idfs.iteritems(): 
             if tfs_key == idfs_key:
                    tfIdfDict[tfs_key] = tfs_value * idfs_value
                    break
    return tfIdfDict
    #return (idfs)
    
recb000hkgj8k = amazonRecToToken.filter(lambda x: x[0] == 'b000hkgj8k').collect()[0][1]
idfsSmallWeights = idfsSmall.collectAsMap()
rec_b000hkgj8k_weights = tfidf(recb000hkgj8k, idfsSmallWeights)

print 'Amazon record "b000hkgj8k" has tokens and weights:\n%s' % rec_b000hkgj8k_weights


# In[68]:

# TEST Implement a TF-IDF function (2f)
Test.assertEquals(rec_b000hkgj8k_weights,
                   {'autocad': 33.33333333333333, 'autodesk': 8.333333333333332,
                    'courseware': 66.66666666666666, 'psg': 33.33333333333333,
                    '2007': 3.5087719298245617, 'customizing': 16.666666666666664,
                    'interface': 3.0303030303030303}, 'incorrect rec_b000hkgj8k_weights')


