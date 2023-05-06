"""
Language Modeling Project
Name:
RollNumber:
"""
import matplotlib.pyplot as plt
import decimal
import hw6_language_tests as test

project = "Language"  # don't edit this


### Stage 1 ###

def loadBook(filename):
    f = open(filename, 'r')
    sentences = []
    for line in f.readlines():
        sentences.append(line.split())
    return sentences


def getCorpusLength(corpus):
    no_of_unigrams = 0
    for sentence in corpus:
        no_of_unigrams += len(sentence)
    return no_of_unigrams


def buildVocabulary(corpus):
    vocabulary = []
    for sentence in corpus:
        for word in sentence:
            if word not in vocabulary:
                vocabulary.append(word)
    return vocabulary


def countUnigrams(corpus):
    data = {}
    for sentence in corpus:
        for word in sentence:
            if word not in data:
                data[word] = 1
            else:
                data[word] += 1
    return data


def getStartWords(corpus):
    start_words = []
    for sentence in corpus:
        start = sentence[0]
        if start not in start_words:
            start_words.append(start)
    # print(start_words)
    # print(len(start_words))
    return start_words


def countStartWords(corpus):
    start_words_data = {}
    for sentence in corpus:
        start = sentence[0]
        if start not in start_words_data:
            start_words_data[start] = 1
        else:
            start_words_data[start] += 1
    # print(len(start_words_data))
    return start_words_data


def countBigrams(corpus):
    bigram_count = {}
    for sentence in corpus:
        sen_length = len(sentence)
        for j in range(sen_length - 1):
            word = sentence[j]
            if word not in bigram_count:
                bigram_count[word] = {}
            temp = bigram_count[word]
            next_word = sentence[j + 1]
            if next_word not in temp:
                temp[next_word] = 0
            temp[next_word] += 1
            bigram_count[word] = temp
    return bigram_count


### Stage 2 ###

def buildUniformProbs(unigrams):
    uniform_probs = []
    prob = 1 / len(unigrams)  # probability of a unigram occuring from the list unigrams

    for value in unigrams:
        uniform_probs.append(prob)
    return uniform_probs


def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    result = []
    for word in unigrams:
        word_count = unigramCounts[word]
        result.append( word_count / totalCount)
    return result


def buildBigramProbs(unigramCounts, bigramCounts):
    # print(len(bigramCounts),len(unigramCounts))
    bigram_probs = {}
    for word in unigramCounts:
        if word in bigramCounts:
            word_count = unigramCounts[word]
            temp = {'words': [], 'probs': []}
            after_words = bigramCounts[word]
            for after_word in after_words:
                after_word_count = after_words[after_word]
                temp['words'].append(after_word)
                temp['probs'].append(after_word_count / word_count)
            bigram_probs[word] = temp
    return bigram_probs


def getTopWords(count, words, probs, ignoreList):
    top_words = {}
    while count != 0:
        maximum = max(probs)
        index = None
        for i in range(len(probs)):
            if probs[i] == maximum:
                index = i
                break
        word = words[index]
        if word not in ignoreList and word not in top_words:
            top_words[word] = maximum
            count = count - 1
        words.pop(index)  # need to pop both maximum prob and associated word
        probs.pop(index)  # because if the word is in ignorelist it won't be removed
        # and the max value will keep on repeating the same so we need to remove it
    # print(top_words)
    return top_words


from random import choices


# The choices() function randomly selects elements from a given sequence, with
# replacement, and returns a list of the selected elements. The weights parameter is an
# optional sequence of weights that assigns a probability to each element in the input
# sequence. Elements with higher weights are more likely to be selected.

def generateTextFromUnigrams(count, words, probs):
    sentence = ""
    while count != 0:
        word = choices(words, weights=probs)[0]
        sentence += word
        sentence += " "
        count = count - 1
    # print(sentence)
    return sentence


def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    sentence = ""
    last_word = None
    while count != 0:
        if sentence == "" or last_word == ".":
            word = choices(startWords, weights=startWordProbs)[0]
            last_word = word
        else:
            temp = bigramProbs[last_word]
            word = choices(temp['words'], weights=temp['probs'])[0]
            last_word = word
        sentence += word
        sentence += " "
        count = count - 1
    return sentence


### Stage 3 ###

ignore = [",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
          "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
          "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
          "that", "onto", "from", "not", "into"]


def graphTop50Words(corpus):
    unigrams = buildVocabulary(corpus)
    unigram_counts = countUnigrams(corpus)
    unigrams_probs = buildUnigramProbs(unigrams, unigram_counts, sum(unigram_counts.values()))
    top_50_unigrams = getTopWords(50, unigrams, unigrams_probs, ignore)
    barPlot(top_50_unigrams, "Top50Words")
    


def graphTopStartWords(corpus):
    start_words = getStartWords(corpus)
    start_words_count = countStartWords(corpus)
    start_words_probs = buildUnigramProbs(start_words, start_words_count, sum(start_words_count.values()))
    top_50_start_words = getTopWords(50, start_words[::], start_words_probs, ignore)
    barPlot(top_50_start_words, "Top 50 Start Words")

def graphTopNextWords(corpus, word):
    bigrams_count = countBigrams(corpus)
    unigram_count = countUnigrams(corpus)
    bigram_probs = buildBigramProbs(unigram_count,bigrams_count)
    words = bigram_probs[word]['words']
    probs = bigram_probs[word]['probs']
    top_10_next_words = getTopWords(10,words[::],probs[::],ignore)
    barPlot(top_10_next_words, "Top 10 Next Words")
    return


def setupChartData(corpus1, corpus2, topWordCount):
    unigrams1 = buildVocabulary(corpus1)
    unigrams1_count = countUnigrams(corpus1)
    total_count1 = getCorpusLength(corpus1)
    unigrams1_probs = buildUnigramProbs(unigrams1,unigrams1_count,total_count1)
    top1 = getTopWords(topWordCount,unigrams1[::],unigrams1_probs[::],ignore)
    unigrams2 = buildVocabulary(corpus2)
    unigrams2_count = countUnigrams(corpus2)
    total_count2 = getCorpusLength(corpus2)
    unigrams2_probs = buildUnigramProbs(unigrams2[::],unigrams2_count,total_count2)
    top2 = getTopWords(topWordCount,unigrams2[::],unigrams2_probs[::],ignore)
    combined_list = list(top1.keys())
    for value in top2.keys():
        if value not in combined_list:
            combined_list.append(value)
    prob1 = []
    count = 0
    for value in combined_list :
        if value in unigrams1:
            index = unigrams1.index(value)
            prob1.append(unigrams1_probs[index])
        else:
            # print(count)
            count+=1
            prob1.append(0)
        # print(value)

    prob2 = []
    count = 0
    for value in combined_list :
        if value in unigrams2:
            index = unigrams2.index(value)
            # print(count)
            count +=1
            prob2.append(unigrams2_probs[index])
        else:
            prob2.append(0)
    answer = {
        'topWords' : combined_list,
        'corpus1Probs' : prob1,
        'corpus2Probs' : prob2
    }
    return answer

def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    top = setupChartData(corpus1,corpus2,numWords)
    sideBySideBarPlots(top['topWords'],top['corpus1Probs'],top['corpus2Probs'],name1,name2,title)
    return


def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    top = setupChartData(corpus1,corpus2,numWords)
    scatterPlot(top['corpus1Probs'],top["corpus2Probs"],top['topWords'],title)
    return


### Stage 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""


def barPlot(dict, title):
    import matplotlib.pyplot as plt
    names = list(dict.keys())
    values = list(dict.values())
    plt.bar(names, values)
    plt.xticks(names, rotation='vertical')
    plt.title(title)
    plt.show()


"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""


def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt
    x = list(range(len(xValues)))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    pos1 = []
    pos2 = []
    for i in x:
        pos1.append(i - width / 2)
        pos2.append(i + width / 2)
    rects1 = ax.bar(pos1, values1, width, label=category1)
    rects2 = ax.bar(pos2, values2, width, label=category2)
    ax.set_xticks(x)
    ax.set_xticklabels(xValues)
    ax.legend()
    plt.title(title)
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    plt.show()


"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""


def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i],  # this is the text
                     (xs[i], ys[i]),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.title(title)
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)
    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " Stage 1 TESTS " +  "#" * 16 + "\n")
    test.stage1Tests()
    print("\n" + "#"*15 + " Stage 1 OUTPUT " + "#" * 15 + "\n")
    test.runStage1()

    # Uncomment these for Stage 2 ##

    print("\n" + "#"*15 + " Stage 2 TESTS " +  "#" * 16 + "\n")
    test.stage2Tests()
    print("\n" + "#"*15 + " Stage 2 OUTPUT " + "#" * 15 + "\n")
    test.runStage2()

    ## Uncomment these for Stage 3 ##
    print("\n" + "#" * 15 + " Stage 3 OUTPUT " + "#" * 15 + "\n")
    test.runStage3()
