"""
Sample LDA on a toy dataset
"""
import gensim
import string

from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

doc1 = "The electric-car maker Tesla is raising $1.5 billion as it ramps up production of the Model 3 sedan, its first mass-market electric car. The company said on Monday that it planned to offer senior notes due in 2025 and would use the proceeds from the offering to further strengthen its balance sheet during rapid scaling of production for the Model 3."
doc2 = "Elon Musk is an entrepreneur in a bubble. Forced to choose between issuing a bit more of Tesla’s turbocharged stock or tapping the overheated junk-bond market to finance the Model 3 ramp-up, Mr. Musk, the company’s founder, opted for the latter. It raises execution risk for the $60 billion electric-car maker, but not by enough to persuade the chief executive to loosen his grip on the wheel. Tesla has just over $3 billion in cash, but it is burning through roughly $1 billion a quarter as it embarks on one of the most daunting gambits in automotive history: taking production of its mass-market vehicle from zero to 400,000 or more a year in just 18 months."
doc3 = "After a meteoric rise that made it, at least briefly, the most valuable car company in America, Tesla arrived at a moment of truth on Friday night as it delivered the first of its mass-market sedans to their new owners. For a decade, the company has been a manufacturer of high-end electric cars in small numbers. But now, Tesla is aiming at much loftier goals. It wants not only to become a large-scale producer in the suddenly crowded field of battery-powered vehicles but also to lure consumers away from mainstream, gasoline-powered automobiles."
doc4 = "When President Trump announced plans to withdraw the United States from the Paris climate accord, Gov. Dannel Malloy of Connecticut responded that his state would continue its push to reduce its carbon footprint. Yet Connecticut is a surprising laggard when it comes to one obvious way to cut carbon emissions: Consumers are not allowed to buy electric vehicles without a costly middleman. Connecticut is one of at least six states that bans carmakers — including Tesla, the nation’s largest manufacturer of electric vehicles — from opening their own storefronts and selling their cars directly to consumers."
doc5 = "Tesla Motors is in discussions to establish a factory in Shanghai, its first in China, a move that could bolster its efforts in one of its major markets even as it further lifts China’s position as a builder of electric cars. In a statement on Thursday, Tesla said it needed to set up more overseas factories to make cars that customers could afford. Such a strategy is a must in China, which charges steep tariffs for imported cars."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

doc_clean = [clean(doc).split() for doc in doc_complete]
lda = gensim.models.ldamodel.LdaModel
ldamodel = lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))