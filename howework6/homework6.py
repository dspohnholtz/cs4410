
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import re
from pathlib import Path
from textblob import TextBlob
from nltk.corpus import stopwords
from operator import itemgetter
from wordcloud import WordCloud

def toChart(orderedList):
    df = pd.DataFrame(orderedList, columns=['word', 'count'])
    axes = df.plot.bar(x='word', y='count', legend=False)
    plt.gcf().tight_layout()

def toCloud(text):
    mask_image = imageio.imread('mask_oval.png')
    wordcloud = WordCloud(colormap='prism', mask=mask_image, background_color='white')
    wordcloud = WordCloud(max_words=20).generate(text)
    wordcloud = wordcloud.to_file('wordCloud.png')

def wordList(items):
    sorted_items = sorted(items, key=itemgetter(1), reverse=True)
    orderedList = sorted_items[1:21]
    toChart(orderedList)

def stopWords(text):
    stop_words = stopwords('english')
    items = text.word_counts.items()
    items = [item for item in items if item[0] not in stop_words]
    wordList(items)

def toBlob(text):
    blob = TextBlob(text)
    stopWords(blob)

def processText(text):
    toBlob(text)
    toCloud(text)

#Run example using hamlet.txt
hamlet = Path('hamlet.txt').read_text()