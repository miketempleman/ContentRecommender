from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from scipy.sparse.csgraph import _validation
import sklearn.neighbors.typedefs
from sklearn.neighbors import NearestNeighbors

import numpy as np
import operator

import config
from bson import json_util
import json

from datetime import datetime, date, time, timedelta
import itertools as it


class ContentRecommend(object):
    create_date = datetime.utcnow()
    days = 15
    training_end = datetime.utcnow()
    db = None
    n_components = 20  # Number of dimension for TruncatedSVD
    account = ''
    svd = None
    normalizer = None
    svdX = None
    vectorizor = None
    training_docs = None
    threshold = 0.25
    k_means = None
    sil_score = -1.0
    cluster_count = 0
    range_n_clusters = [3, 4, 5, 6, 7, 8]
    missionId = ''

    def __init__(self, mission_id, db_name='plover_development', db_port=27017, db_host='localhost'):
        self.missionId = mission_id
        config.LOGGER.info('Instantiation recommender')
        self.connect(db_name, self.missionId, db_port=db_port, db_host=db_host)

    def connect(self, db_name="plover_development", mission_id="", db_port=27017, db_host='localhost'):
        config.LOGGER.info('Instantiating recommender object for mission %s', mission_id)
        config.LOGGER.debug('Using database %s, host %s and port %s', db_name, db_host, db_port)

        try:
            client = MongoClient(db_host, db_port)
            self.db = client[db_name]
            profile = self.db.socialProfile.find_one({'mission': ObjectId(self.missionId)})
            self.account = self.db.linkedAccount.find_one({'_id': profile['account']})
            if self.account is None:
                config.LOGGER.debug('No such account id')
            self.setup_training()
        except Exception as ex:
            config.LOGGER.error("Error %s opening mission _id=%s", ex.message, self.missionId)


    def get_updates(self, maximum=100, conditions={}):
        documents = []
        config.LOGGER.info('Getting timeline updates for mission %s', self.missionId)
        config.LOGGER.debug(' query condition: %s', json.dumps(conditions, default=json_util.default))
        try:
            if self.account is None:
                config.LOGGER.debug('No account id')
            else:
                projection = {'keywords': 1, 'text': 1, 'externalID': 1, 'postTime': 1, 'sender': 1,
                              'quotedStatus': 1}
                updates = self.db.statusUpdate.find(conditions, projection).sort('postTime', pymongo.DESCENDING).limit(maximum)
                for tw in updates:
                    if 'quotedStatus' in tw:
                        tw['text'] += " QT: " + tw['quotedStatus']['text']
                        for keyword in tw['quotedStatus']['keywords']:
                            tw['keywords'].append(keyword)
                    smu = self.db.socialMediaUser.find_one({'_id': tw['sender']}, {'screenNameLC': 1})
                    if smu is not None:
                        tw['keywords'].append(smu['screenNameLC'])
                    documents.append(tw)


        except Exception as ex:
            config.LOGGER.error("Error %s getting updates from timeline for mission %s", ex.message, self.missionId)

        config.LOGGER.debug('Found %d updates in timeline', len(documents))
        return documents



    def topics(self, n_components, n_out=7, n_weight=5, topic=None):
        config.LOGGER.info('Get topices timeline for %s', self.account['profile']['preferredUsername'])
        results = []
        terms = self.vectorizer.get_feature_names()
        if topic is None:
            for k in range(n_components):
                idx = {i: abs(j) for i, j in enumerate(self.svd.components_[k])}
                sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
                weight = np.mean([item[1] for item in sorted_idx[0:n_weight]])

                for item in sorted_idx[0:n_out - 1]:
                    results.append({'term': terms[item[0]], 'weight': item[1]})
        else:
            m = max(self.svd.components_[topic])
            idx = {i: abs(j) for i, j in enumerate(svd.components_[topic])}
            sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
            weight = np.mean([item[1] for item in sorted_idx[0:n_weight]])

            for item in sorted_idx[0:n_out - 1]:
                results.append({'term': terms[item[0]], 'weight': item[1]})
        results

    def get_componentCount(self, min=.05):
        count = 0
        for k in range(len(self.svd.components_)):
            idx = {i: abs(j) for i, j in enumerate(self.svd.components_[k])}
            sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
            kcount = 0
            for entry in (sorted_idx):
                if entry[1] > min:
                    kcount += 1
                else:
                    break
            if kcount > count:
                count = kcount
        return count

    def setup_training(self, end_time=datetime.utcnow(), days=15, maximum=1000):
        try:
            start = end_time - timedelta(minutes=days*24*60)
            condition = {'missions': ObjectId(self.missionId), '$or': [{'favorited': True}, {'sentByMe': True}],
                        'postTime': {'$gt': start, '$lte': end_time}}
            self.training_docs = self.get_updates(conditions=condition, maximum=1000)
            config.LOGGER.info('Train model for %s', self.account['profile']['preferredUsername'])
            if len(self.training_docs) > 25:
                config.LOGGER.debug('Found %d updates for training from %s', len(self.training_docs),
                                    self.account['profile']['preferredUsername'])
                self.training_end = end_time
                self.days = days

                trainingTokenized = [' '.join(doc['keywords']) for doc in self.training_docs]
                self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=500, use_idf=True,
                                                  strip_accents='ascii')
                X = self.vectorizer.fit_transform(trainingTokenized)
                if X.shape[1] <= self.n_components:
                    self.n_components = X.shape[1] - 1
                config.LOGGER.debug('%d components found for  SVD', self.n_components)
                self.svd = TruncatedSVD(self.n_components, algorithm='arpack')
                self.svdX = self.svd.fit_transform(X)
                # self.n_components = self.get_componentCount(self.threshold)
                # self.svd = TruncatedSVD(self.n_components, random_state=10)
                # self.svdX = self.svd.fit_transform(X)
                self.normalizer = Normalizer().fit(self.svdX)
                self.svdX = self.normalizer.transform(self.svdX)

                # Clustering
                config.LOGGER.debug('Determining cluster count ')
                for n_clusters in self.range_n_clusters:
                    self.k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10,
                                          verbose=False, random_state=10)
                    self.k_means.fit(self.svdX)
                    score = metrics.silhouette_score(self.svdX, self.k_means.labels_)
                    if score > self.sil_score:
                        self.sil_score = score
                        self.cluster_count = n_clusters

                config.LOGGER.debug('Cluster count is %d, Silhouette Coefficient is %0.3f  ', self.cluster_count,
                                    self.sil_score)
                self.k_means = KMeans(n_clusters=self.cluster_count, init='k-means++', max_iter=100, n_init=4,
                                      verbose=False, random_state=10)
                self.k_means.fit(self.svdX)

                # now get the top tweets for each cluster
                x_transform = self.k_means.transform(self.svdX)
                x_predict = self.k_means.predict(self.svdX)

                self.all_cluster_dist = []
                for i in range(self.cluster_count):
                    cluster_distance = []
                    for j in range(len(x_predict)):
                        if x_predict[j] == i and sum(self.svdX[j]) != 0.0:
                            cluster_distance.append(
                                {'index': j, 'cluster': i, 'dist': np.sqrt(sum([y * y for y in x_transform[j]]))})
                    newlist = sorted(cluster_distance, key=operator.itemgetter('dist'), reverse=False)
                    self.all_cluster_dist.append(newlist)

                #now verify this
                self.self_test()

            else:
                config.LOGGER.info('Too few training updates from user timeline')
                self.svd = None
        except Exception as ex:
            config.LOGGER.exception("Error %s computing SVD and kmeans from user history for mission %s", ex.message,
                                self.missionId)


    def self_test(self):
        try:
            config.LOGGER.info("Beginning self test. Better if it were cross validation but not enough data for that")
            results = self.find_recommendations(self.training_docs, top=10, quality=.001, min_examples=1)
            config.LOGGER.info("Self test found %d recommendations", len(results))
            for rec in results:
                if rec['text'] != rec['samples_svd'][0]:
                    config.LOGGER.error("Error training SVD for mission %s in tweet %s", self.missionId, rec['text'])
        except Exception as ex:
            config.LOGGER.error("Error in self test building training for mission %s", ex.message, self.missionId)


    def find_recommendations(self, tweets=[], top=10, quality=.1, min_examples=1):

        working_size = top * 2
        working_list = []
        result_list = []
        try:
            config.LOGGER.info('Generating content recommendations for user %s',
                               self.account['profile']['preferredUsername'])
            if self.svd is not None:
                if len(tweets) < top:
                    config.LOGGER.debug("Too few tweets passed for recommendation")
                    return []

                tokenized_tweets = [' '.join(doc['keywords']) for doc in tweets]
                Y = self.vectorizer.transform(tokenized_tweets)
                svdY = self.svd.transform(Y)
                svdY = self.normalizer.transform(svdY)
                y_transform = self.k_means.transform(svdY)
                # terms = self.vectorizer.get_feature_names()

                selected_updates = []
                y_predict = self.k_means.predict(svdY)

                for i in range(self.cluster_count):
                    cluster_distance = []
                    for j in range(len(y_predict)):
                        if y_predict[j] == i and sum(svdY[j]) != 0.0:
                            cluster_distance.append(
                                {'index': j, 'cluster': i, 'dist': np.sqrt(sum([y * y for y in y_transform[j]]))})
                    newlist = sorted(cluster_distance, key=operator.itemgetter('dist'), reverse=False)
                    selected_updates.append(newlist)

                temp = [entry for entry in it.izip_longest(*selected_updates)]
                clean_list = filter(lambda x: x is not None, [entry for tuple in temp for entry in tuple])[0:top]
                clean_list_svdY = [svdY[entry['index']] for entry in clean_list]
                config.LOGGER.debug("Found %i possible matches in topic clusters " % len(clean_list_svdY))

                neigh = NearestNeighbors()
                neigh.fit(self.svdX)
                if len(clean_list_svdY) > 0:
                    distances, svd_neighbors = neigh.radius_neighbors(X=clean_list_svdY, radius=quality)
                else:
                    svd_neighbors =[]

                examples=[]
                for idx, entry in enumerate(svd_neighbors):
                    if len(entry) >= min_examples:
                        config.LOGGER.debug("Suggested tweet has %d examples" % len(entry))
                        original = tweets[clean_list[idx]['index']]['text']
                        for jdx, neighbor in enumerate(entry):
                            examples.append({'text':self.training_docs[neighbor]['text'], 'dist':distances[idx][jdx]})
                        sorted_examples = sorted(examples, key=operator.itemgetter('dist'), reverse=False)
                        min_examples = [item['text'] for item in sorted_examples][:min_examples]
                        t1 = self.training_docs[self.all_cluster_dist[clean_list[idx]['cluster']][0]['index']]['text']
                        t2 = self.training_docs[self.all_cluster_dist[clean_list[idx]['cluster']][1]['index']]['text']
                        working_list.append({"dist": sorted_examples[0]['dist'], "text": original,
                                                     "id": str(tweets[clean_list[idx]['index']]['_id']),
                                                     "sender": str(tweets[clean_list[idx]['index']]['sender']),
                                                     'samples_svd': min_examples, 'samples_cluster':[t1,t2]})

                result_list = sorted(working_list, key=operator.itemgetter('dist'), reverse=False)
            return result_list[:top]

        except Exception as ex:
            config.LOGGER.error("Error %s computing recommendations for mission %s", ex.message, self.missionId)
            return []

    def recommend_from_timeline(self, end_time=datetime.utcnow(), minutes_prior=15, top=10, quality=.1, min_examples=1):

        try:
            config.LOGGER.info("generating content recommendation from timeline for %s" % self.account['profile']['preferredUsername'])
            results = []
            if self.svd is not None:
                start = end_time - timedelta(minutes=minutes_prior)
                condition = {'missions': ObjectId(self.missionId), '$or': [{'favorited': False}, {'sentByMe': False}, {'mentionsMe' : False},{'retweetOfMe':False}],
                            'postTime': {'$gt': start, '$lte': end_time}}
                tweets = self.get_updates(maximum=1000, conditions=condition)
                config.LOGGER.debug('%d updates from account timeline read from database', len(tweets))
                results = self.find_recommendations(tweets, top=top, quality=quality, min_examples=min_examples)
                config.LOGGER.debug('%d recommendations found for mission %s', len(tweets), self.missionId)
            return results[:top]

        except Exception as ex:
            config.LOGGER.error("Error %s computing recommendations for mission %s", ex.message, self.missionId)
            return []
