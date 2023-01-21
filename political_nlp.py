from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from textblob import TextBlob
import nltk
import itertools
#nltk.download('brown')
#nltk.download('punkt')


class pnlp:

    def __init__(self, filename, negatives):

        # take in file name and list of negative words

        # construct pnlp vals
        self.data = filename
        self.neg_words = negatives
        self.neg_ratio = 0
        self.nouns = []
        self.pos = 0
        self.neg = 0
        self.neu = 0
        self.text = ''
        self.word_length = 0
        self.noun_scores = {}

    @staticmethod
    def _nyt_parser(self):
        """ parser for dirty nyt json files

        :param self: self
        :return: dct (dict): dct with cleaned articles
        """

        # open file and use json.load
        f = open(self.data)
        raw = json.load(f)

        # store all k/v pairs where text length is 0
        del_lst = []
        for k, v in raw.items():
            if len(v) == 0:
                del_lst.append(k)

        # delete all fields of 0
        for i in range(len(del_lst)):
            del raw[del_lst[i]]

        # iterate through each of the articles
        for k, v in raw.items():

            # get article
            article = v

            # start after lyrics header
            article = article.replace("\\", "")

            raw[k] = article

        return raw

    @staticmethod
    def _nyp_parser(self):
        """ parser for dirty nyp json files

        :param self: self
        :return: dct (dict): dct with cleaned articles
        """

        # open file and use json.load
        f = open(self.data)
        raw = json.load(f)

        # store all k/v pairs where text length is 0
        del_lst = []
        for k, v in raw.items():
            if len(v) == 0:
                del_lst.append(k)

        # delete all fields of 0
        for i in range(len(del_lst)):
            del raw[del_lst[i]]

        # iterate through each of the articles
        for k, v in raw.items():

            # get article
            article = v

            # start after lyrics header
            article = article.replace("<strong>", "").replace("</strong>", "").replace("<em>", "").replace("</em>", "")
            article = article.replace("</a>", "")
            article = article.replace("\n","").replace("*","")

            # if article has href section, clean
            if "<a href=" in article:
                s = article.split("<a href=")

                # add all cleaned strings
                str = ''
                for i in range(1, len(s)):
                    after = s[i].split(">")
                    str += " ".join(after[1:])

                # set article to removed href version
                article = str

            # redefine k,v pair
            raw[k] = article

        # return dict
        return raw

    def load_text(self, parser="nyt"):
        """ registers the text file with the NLP framework
        :param parser (string):  specifies if file is nyp or nyt
        :return: none
        """

        # default parser
        if parser == "nyp":
            dct = pnlp._nyp_parser(self)

        # json parser
        elif parser == "nyt":
            dct = pnlp._nyt_parser(self)

        # call save results depending on list of strings of words from parsers
        self.text = dct

        # save cleaned dict of texts
        self._save_results(dct)

    def _save_results(self, dct):
        """ saves sentiment scores for year/news source

        :param results (lst): list of strings of phrases or full song lyrics
        :return: none
        """

        # make lists for 3 Vader sentiment scores
        pos = []
        neg = []
        neu = []

        # list for negative ratio score
        neg_rat = []

        # list of nouns
        noun_lst = []

        # list for length of each text
        lengths = []

        # iterate through each article
        for k, v in dct.items():

            # Create a SentimentIntensityAnalyzer object.
            sid_obj = SentimentIntensityAnalyzer()

            # polarity_scores method of SentimentIntensityAnalyzer
            # object returns sentiment dict with positivity, negativity, and neutrality
            sentiment_dict = sid_obj.polarity_scores(v)

            # append 3 vader scores
            pos.append(sentiment_dict['pos'])
            neg.append(sentiment_dict['neg'])
            neu.append(sentiment_dict['neu'])

            # set count of neg words to 0
            count = 0

            # iterate through each word in article
            for word in v.split(" "):

                # add 1 to count if there is negative word
                if word in self.neg_words:
                    count += 1

            # append to neg_ratio each negative word ratio
            # calculated by dividing count of negative words and total count of words in article
            neg_rat.append(count/len(v.split(" ")))

            # append length of article
            lengths.append(len(v.split(" ")))

            # make text blob object and append all nouns for article with noun_phrases function
            blob = TextBlob(v)
            noun_lst.append(list(blob.noun_phrases))

        # set word count length for year by summing length of all articles
        self.word_length = sum(lengths)

        # save noun list
        self.nouns = noun_lst

        # save average negative ratio for all articles of a year
        self.neg_ratio = sum(neg_rat)/len(neg_rat)

        # save articles' average sentiment scores
        self.pos = sum(pos) / len(pos)
        self.neg = sum(neg) / len(neg)
        self.neu = sum(neu) / len(neu)

        # make nouns for all articles into one list
        nouns = list(itertools.chain.from_iterable(noun_lst))

        # make dict for noun counts
        noun_dct = {}

        # iterate through all nouns
        for word in nouns:

            # clean nouns
            if "<" not in word and ">" not in word and word != '’ s' and '"' not in word:

                # add new nouns to dict
                if word not in noun_dct:
                    noun_dct[word] = 1

                # add 1 to count of preexisting nouns
                else:
                    noun_dct[word] += 1

        # save sorted noun score dictionary
        self.noun_scores = dict(sorted(noun_dct.items(), key=lambda item: item[1], reverse=True))
