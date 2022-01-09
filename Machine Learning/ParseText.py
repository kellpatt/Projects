import numpy as np
from nltk.stem.snowball import SnowballStemmer  # http://www.nltk.org/howto/stem.html
import codecs


class ParseText:
    def __init__(self, filename, stop_words):
        self.stop_words = stop_words
        self.vocabulary = {}
        self.text = []
        self.target = None #Ex: "spam" or "ham"
        self.filename = None
        self.loadData(filename)

    def toString(self, array):
        array = [int(x) for x in array]
        return array

    def parseLine(self, line):
        words = [x.strip('\n') for x in line]
        #words = [x.strip('\r') for x in line]
        # CHECK: 1705 spam file after changing strips

        # remove non-ascii characters
        ascii_words = []
        for word in words:
            valid = True
            #print("word =", word)

            if len(word) > 0:
                for idx, char in enumerate(word):
                    #print("len(word) =", len(word))
                    #print("idx =",idx)
                    if (ord(char) < 65) or (ord(char) > 90 and ord(char) < 97) or (ord(char) > 122):
                        #print("found non-ascii")
                        #print("ord(char) = ", ord(char))

                        # Allow words (stripped) that have a non-ascii character only at begin or end of word
                        if (idx != 0) and (idx != len(word)-1):
                            valid = False
                        else:
                            word = word.strip(char)
                            #print(len(word.strip(char)))
                            #print("word w/o char = ", word)

                if valid == True:
                    if len(word) > 0: # word after stripping
                        ascii_words.append(word)

        #print("ascii words = ", ascii_words)
        return ascii_words

    def loadData(self, fname):
        self.text = []
        # Strip utf-8 encoded character errors
        with codecs.open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_text = line.split(' ')
                line_text = self.parseLine(line_text)
                self.text.append(line_text)

        # flatten text list to 1D
        self.text = [element for sublist in self.text for element in sublist]

        if self.stop_words == False:
            self.text = self.removeStopWords(self.text, 'stop_words.txt')

        self.text = self.removePlurals(self.text)

        # calculate frequency of all words in document, assign to unique vocabulary
        for word in self.text:
            self.vocabulary[word] = np.sum(np.array(self.text) == word)  # set key value to 0

        if self.stop_words == False:
            self.text = self.removeStopWords(self.text, 'stop_words.txt')

        #self.vocabulary = set(self.text)


    def removeStopWords(self, words, fname):
        stop_words = []
        with open(fname) as f:
            stop_words = [line.strip('\n') for line in f]

        new_words = [x for x in words if x not in stop_words]
        return new_words

    def removePlurals(self, words):
        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word) for word in words]
        return words

if __name__ == "__main__":
    training_data = ParseText('text.txt', stop_words=True)  # train/ham/0004.1999-12-14.farmer.ham.txt')