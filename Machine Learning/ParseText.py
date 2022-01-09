import numpy as np
from nltk.stem.snowball import SnowballStemmer  # http://www.nltk.org/howto/stem.html
import codecs


class ParseText:
    def __init__(self, filename, stop_words):
        self.stop_words = stop_words
        self.vocabulary = {}
        self.text = []
        self.target = None  # Ex: "spam" or "ham"
        self.filename = None
        self.load_data(filename)

    @staticmethod
    def to_string(array):
        array = [int(x) for x in array]
        return array

    @staticmethod
    def parse_line(line):
        words = [x.strip('\n') for x in line]
        # words = [x.strip('\r') for x in line]

        # remove non-ascii characters
        ascii_words = []
        for word in words:
            valid = True
            if len(word) > 0:
                for idx, char in enumerate(word):
                    if (ord(char) < 65) or (ord(char) > 90 and ord(char) < 97) or (ord(char) > 122):
                        # Allow words w/ non-ascii character only at begin or end of word
                        if (idx != 0) and (idx != len(word) - 1):
                            valid = False
                        else:
                            word = word.strip(char)

                if valid is True:
                    if len(word) > 0:  # word after stripping
                        ascii_words.append(word)

        return ascii_words

    def load_data(self, fname):
        self.text = []
        # Strip utf-8 encoded character errors
        with codecs.open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_text = line.split(' ')
                line_text = self.parse_line(line_text)
                self.text.append(line_text)

        # flatten text list to 1D
        self.text = [element for sublist in self.text for element in sublist]

        if self.stop_words is False:
            self.text = self.remove_stopwords(self.text, 'stop_words.txt')

        self.text = self.remove_plurals(self.text)

        # calculate frequency of all words in document, assign to unique vocabulary
        for word in self.text:
            self.vocabulary[word] = np.sum(np.array(self.text) == word)  # set key value to 0

        if self.stop_words is False:
            self.text = self.remove_stopwords(self.text, 'stop_words.txt')

    @staticmethod
    def remove_stopwords(words, fname):
        stop_words = []
        with open(fname) as f:
            stop_words = [line.strip('\n') for line in f]

        new_words = [x for x in words if x not in stop_words]
        return new_words

    @staticmethod
    def remove_plurals(words):
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word) for word in words]
        return words


if __name__ == "__main__":
    training_data = ParseText('text.txt', stop_words=True)
