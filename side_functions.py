import nltk
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sn
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score # tp      / ( tp + fp )
from sklearn.metrics import accuracy_score # tp + tn / ( tp + fp + tn + fn  )
from sklearn.metrics import recall_score    # tp      / ( tp + fn )
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import TweetTokenizer
from sklearn.model_selection import RandomizedSearchCV
import pyLDAvis.gensim
from gensim.models import LdaModel
from gensim import corpora
import nltk
from string import punctuation
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
pyLDAvis.enable_notebook()


# nltk.download('stopwords')
dict_score = {'precision': precision_score, 'accuracy': accuracy_score,
              'recall score': recall_score,
              'confusion_matrix': confusion_matrix}


def get_train_test_data(find_and_concatenate_expressions=False):

    def remove_url(tokens):
        tokens = filter(lambda x: "http" not in x, tokens)
        return list(tokens)

    def remove_hashtags(tokens):
        tokens = map(lambda x: x.replace('#', ''), tokens)
        return list(tokens)

    db = pd.read_excel("Classeur1.xlsx", encoding="utf-8")
    dict_values = {'Not Relevant': -1, 'Relevant': 1, "Can't Decide": 0}
    db["to_predict"] = db.choose_one.map(dict_values)
    db = db[["text", "to_predict"]]
    twtk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    db["token_retreated_text"] = db["text"].apply(lambda x: remove_hashtags(remove_url(twtk.tokenize(x))))
    db["retreated_text"] = db["token_retreated_text"].apply(lambda x: " ".join(x))

    if find_and_concatenate_expressions:
        db["token_retreated_text"] = clean_corpus(db["retreated_text"])
        db["retreated_text"] = db["token_retreated_text"].apply(lambda x: " ".join(x))

    msk = np.random.rand(len(db)) < 0.8
    train = db[msk]
    test = db[~msk]

    return train, test


def tokenize_sentences(corpus):
    twtk = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokenized_sentences = corpus.apply(twtk.tokenize)

    return tokenized_sentences


def show_phrases(corpus, threshold=1000, shown=1000):
    # Training the multi-word expression detector
    tokenized_sentences = tokenize_sentences(corpus)
    phrases = Phrases(tokenized_sentences, threshold=threshold)
    i = 0
    for phrase, score in phrases.export_phrases(tokenized_sentences):
        if i > shown:
            break
        else:
            print("Expression : {0}, score = {1}".format(phrase.decode('utf-8'), score))
        i = i + 1


def clean_corpus(corpus, threshold=1000):
    tokenized_sentences = tokenize_sentences(corpus)
    phrases = Phrases(tokenized_sentences.values.tolist(), threshold=threshold)

    # This lets you use it with less RAM and faster processing.
    # But it will no longer be possible to update the detector with new training
    # samples
    phraser = Phraser(phrases)

    # Merging multi-word expressions in the tokenization
    clean_corpus = []
    for sentence in tokenized_sentences:
        clean_corpus.append(phraser[sentence])

    return clean_corpus


def get_lda_model(train):
    nltk.download('stopwords')
    en_stop = set(nltk.corpus.stopwords.words('english'))
    to_be_removed = list(en_stop) + list(punctuation)

    tok = TweetTokenizer()
    # Tokenizing + removing stopwords
    text_data = list(train.retreated_text.apply(lambda x: list(filter(lambda a: a.lower() not in to_be_removed,
                                                                      tok.tokenize(x)))).array)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    ldamodel = LdaModel(corpus, id2word=dictionary, num_topics=4)
    return corpus, dictionary, ldamodel


def display_lda_model(corpus, dictionary, ldamodel):
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.display(lda_display)
    # pyLDAvis.show(lda_display)


def train_on_dataset(train, option="naive_bow", w2v_model=None):

    y_train = train["to_predict"].values

    # Initialize the "vectorizer" object, which is our bag of words tool
    # Creates the vectorizer and vectorizes the texts
    # We convert out vector to an array because it's easier, and faster to work with

    estimator = lgb.LGBMClassifier(num_leaves=31, scoring='accuracy')

    param_grid = {'max_depth': [5, 10, 20, 50, 100],
                  'min_data_in_leaf': [5, 10, 20, 40],
                  'learning_rate': [0.01, 0.1, 1],
                  'n_estimators': [10, 20, 40, 100, 150, 200, 250]}

    gbm = RandomizedSearchCV(estimator, param_grid, n_iter=20, cv=3, verbose=1)

    if option == "naive_bow":
        x_train = train["retreated_text"].values
        vectorizer = TfidfVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     # ngram_range=(1, 2),
                                     # We take words but also couples of words
                                     token_pattern=r'\b\w+\b',
                                     max_features=5000)
        pipe = Pipeline([('tfidf', vectorizer), ('gbm', gbm)])
    elif option == "w2v":
        x_train = word2vec_features(train["token_retreated_text"], w2v_model)
        pipe = Pipeline([('gbm', gbm)])
    else:
        pipe = None
        x_train = None
    pipe.fit(x_train, y_train)
    return pipe


def test_on_dataset(pipe, test, option="naive_bow", w2v_model=None):

    if option == "naive_bow":
        x_test = test["retreated_text"].values
    elif option == "w2v":
        x_test = word2vec_features(test["token_retreated_text"], w2v_model)
    else:
        x_test = None

    y_test = test["to_predict"].values
    y_pred = pipe.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    show_eval(y_test, y_pred)


def train_word2vec(train):
    cpu = cpu_count()
    w2v = Word2Vec(train, size=100, window=5, min_count=3, workers=cpu)
    w2v.train(train, total_examples=len(train), epochs=10)
    return w2v


def get_vect(word, model):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros((model.vector_size,))


def sum_vectors(phrase, model):
    return sum(get_vect(w, model) for w in phrase)


def word2vec_features(X, model):
    feats = np.vstack([sum_vectors(p, model) for p in X])
    return feats


def show_eval(y_true, y_pred):
    '''
    Show eval metrics.  Takes binarized y true and pred along with trained binarizer for label names
    '''
    y_dict = {-1: "Pas de désastre", 0: "Impossible de trancher", 1: "Désastre"}
    y_true_names = pd.Series(y_true).map(y_dict)
    y_pred_names = pd.Series(y_pred).map(y_dict)
    print(classification_report(y_true_names, y_pred_names))
    cm = confusion_matrix(y_true_names, y_pred_names)
    labels = ["Pas de désastre", "Impossible de trancher", "Désastre"]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    # config plot sizes
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 18}, cmap='coolwarm', linewidth=0.5, fmt="")
    plt.show()
    y_true = pd.get_dummies(y_true_names).values
    y_pred = pd.get_dummies(y_pred_names)
    y_pred["Impossible de trancher"] = 0
    y_pred = y_pred.values

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, label in enumerate(labels):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(label, roc_auc[i])

    for i, label in enumerate(labels):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for '+label)
        plt.legend(loc="lower right")
        plt.show()


def compute_tsne(train, w2v_model):
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    x_train = word2vec_features(train["token_retreated_text"], w2v_model)
    tsne_results = tsne.fit_transform(x_train)
    return tsne_results


def plot_tsne(train, tsne_results):
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['label'] = train["to_predict"]
    df_subset["mask"] = True

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
#        palette=sns.color_palette("hls", 2),  # len(set(all_keywords))),
        data=df_subset,
        legend="full",
        alpha=0.3
    )



