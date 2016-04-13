
from pymongo import MongoClient
import pymongo
from bson.objectid import ObjectId

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans

from scipy.sparse.csgraph import _validation
import sklearn.neighbors.typedefs

import numpy as np
import operator

import config

from datetime import datetime, date, time, timedelta

class ContentRecommend(object):

    create_date=datetime.utcnow()
    days=15
    training_end = datetime.utcnow()
    db=None
    n_components = 20     # Number of dimension for TruncatedSVD
    account = ''
    svd=None
    svdX=None
    vectorizor=None
    training_docs=None
    threshold=0.25
    k_means=None
    sil_score=-1.0
    cluster_count=0
    range_n_clusters = [3, 4, 5, 6,7,8]
    missionId = ''

    def __init__(self, mission_id, db_name='plover_development', db_port=27017, db_host='localhost'):
        self.missionId=mission_id
        config.LOGGER.info('Instantiation recommender')
        self.connect(db_name, self.missionId, db_port=db_port, db_host=db_host)

    def connect(self, db_name="plover_development", mission_id="", db_port=27017, db_host='localhost'):
        config.LOGGER.info('Instantiating recommender object for mission %s', mission_id)
        config.LOGGER.debug('Using database %s, host %s and port %s', db_name, db_host, db_port)

        try:
            client = MongoClient(db_host, db_port)
            self.db=client[db_name]
            profile=self.db.socialProfile.find_one({'mission':ObjectId(self.missionId)})
            self.account = self.db.linkedAccount.find_one({'_id': profile['account']})
            if self.account is None:
                config.LOGGER.debug('No such account id')
            self.setup_training()
        except Exception as ex:
            config.LOGGER.error("Error %s opening mission _id=%s", ex['msg'], self.missionId)

    def get_input_updates(self, end_time=datetime.utcnow(), minutes=15, maximum=100):
        start=end_time-timedelta(minutes=minutes)
        documents=[]
        config.LOGGER.info('Getting timeline updates from %s to %s', str(start), str(end_time))
        try:
            if self.account is None:
                config.LOGGER.debug('No account id')
            else:
                profile = self.db.socialProfile.find_one({'account': self.account['_id']})
                if profile is None:
                    config.LOGGER.debug('Failed to find profile id %s', str(self.account['_id']))
                else:
                    condition   = {'missions':profile['mission'], 'inReplyToID':-1, 'retweet': False, 'sentByMe':False, 'favorited':False, 'mentionsMe':False, 'postTime':{'$gt':start, '$lte':end_time}}
                    projection = {'keywords':1, 'text':1, 'externalID':1, 'postTime':1, 'sender':1, 'quotedStatus':1 }
                    updates=self.db.statusUpdate.find(condition, projection).sort('postTime', pymongo.DESCENDING)
                    for tw in updates:
                        if 'quotedStatus' in tw:
                            tw['text'] += " QT: " + tw['quotedStatus']['text']
                            for keyword in tw['quotedStatus']['keywords']:
                                 tw['keywords'].append(keyword)
                        smu=self.db.socialMediaUser.find_one({'_id':tw['sender']},{'screenNameLC':1})
                        if smu is not None:
                            tw['keywords'].append(smu['screenNameLC'])
                        documents.append(tw)


        except Exception as ex:
            config.LOGGER.error("Error %s getting updates from timeline for mission %s", ex['msg'], self.missionId)

        config.LOGGER.debug('Found %d updates in timeline', len(documents))
        return documents



    def get_training_updates(self, end_time=datetime.utcnow(), days=15):
        '''
            Retrieves the tokenized version of the timelines,
            followers of a 'parent' account
            that we choose to include in the corpus: is_included: True
        '''
        config.LOGGER.info('Getting training updates from timeline for %s', self.account['profile']['preferredUsername'])
        start=end_time-timedelta(days=days)
        documents=[]
        try:
            if self.account is not None:
                profile = self.db.socialProfile.find_one({'account': self.account['_id']})
                if profile is not None:
                    condition   = {'missions':profile['mission'], '$or':[{'favorited':True}, {'sentByMe':True}],'postTime':{'$gt':start,'$lte': end_time}}
                    projection = {'keywords':1, 'text':1, 'externalID':1, 'postTime':1, 'quotedStatus':1}
                    updates=self.db.statusUpdate.find(condition, projection)
                    for tw in updates:
                        if 'quotedStatus' in tw:
                            tw['text'] += " QT " + tw['quotedStatus']['text']
                            for keyword in tw['quotedStatus']['keywords']:
                                 tw['keywords'].append(keyword)
                        documents.append(tw)

            return documents
        except Exception as ex:
            config.LOGGER.error("Error %s getting training updates from user history for mission %s", ex['msg'], self.missionId)
            return []

    def topics(self, n_components, n_out = 7, n_weight = 5, topic = None):
        config.LOGGER.info('Get topices timeline for %s', self.account['profile']['preferredUsername'])
        results=[]
        terms=self.vectorizer.get_feature_names()
        if topic is None:
            for k in range(n_components):
                idx = {i:abs(j) for i, j in enumerate(self.svd.components_[k])}
                sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
                weight = np.mean([ item[1] for item in sorted_idx[0:n_weight] ])

                for item in sorted_idx[0:n_out-1]:
                    results.append({'term':terms[item[0]],'weight':item[1]})
        else:
            m = max(self.svd.components_[topic])
            idx = {i:abs(j) for i, j in enumerate(svd.components_[topic])}
            sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
            weight = np.mean([ item[1] for item in sorted_idx[0:n_weight] ])

            for item in sorted_idx[0:n_out-1]:
                results.append({'term':terms[item[0]],'weight':item[1]})
        results



    def get_componentCount(self, min=.1):
        count=0
        for k in range(len(self.svd.components_)):
            idx = {i:abs(j) for i, j in enumerate(self.svd.components_[k])}
            sorted_idx = sorted(idx.items(), key=operator.itemgetter(1), reverse=True)
            kcount=0
            for entry in (sorted_idx):
                if entry[1] > min :
                    kcount += 1
                else:
                    break
            if kcount > count:
                count = kcount
        return count


    def setup_training(self, end_time=datetime.utcnow(), days=15):
        try:
            self.training_docs = self.get_training_updates(end_time, days)
            config.LOGGER.info('Train model for %s', self.account['profile']['preferredUsername'])
            if len(self.training_docs) > 25:
                config.LOGGER.debug('Found %d updates for training from %s', len(self.training_docs), self.account['profile']['preferredUsername'])
                self.training_end = end_time
                self.days = days

                trainingTokenized   = [ ' '.join(doc['keywords']) for doc in self.training_docs]
                self.vectorizer  = TfidfVectorizer(max_df=0.9,  min_df=2, max_features=500, use_idf=True, strip_accents='ascii')
                X = self.vectorizer.fit_transform(trainingTokenized)
                if X.shape[1] <= self.n_components:
                    self.n_components = X.shape[1]-1
                config.LOGGER.debug('%d components found for  SVD', self.n_components)
                self.svd  = TruncatedSVD(self.n_components, random_state= 10)
                self.svdX = self.svd.fit_transform(X)
                self.n_components = self.get_componentCount(self.threshold)
                self.svd  = TruncatedSVD(self.n_components, random_state= 10)
                self.svdX = self.svd.fit_transform(X)
                # Clustering
                config.LOGGER.debug('Determining cluster count ')
                for n_clusters in self.range_n_clusters:
                    self.k_means = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10,
                                verbose=False, random_state= 10)
                    self.k_means.fit(self.svdX)
                    score = metrics.silhouette_score(self.svdX, self.k_means.labels_)
                    if score > self.sil_score:
                        self.sil_score = score
                        self.cluster_count = n_clusters

                config.LOGGER.debug('Cluster count is %d, Silhouette Coefficient is %0.3f  ', self.cluster_count, self.sil_score)
                self.k_means = KMeans(n_clusters=self.cluster_count, init='k-means++', max_iter=100, n_init=4,
                        verbose=False, random_state= 10)
                self.k_means.fit(self.svdX)

                #now get the top tweets for each cluster
                x_transform = self.k_means.transform(self.svdX)
                x_predict = self.k_means.predict(self.svdX)

                self.all_cluster_dist = []
                for i in range(self.cluster_count):
                    cluster_distance=[]
                    for j in range(len(x_predict)):
                        if x_predict[j] == i and sum(self.svdX[j]) != 0.0:
                            cluster_distance.append({'index':j, 'cluster': i, 'dist':np.sqrt(sum([y*y for y in x_transform[j]]))})
                    newlist=sorted(cluster_distance, key=operator.itemgetter('dist'), reverse=False)
                    self.all_cluster_dist.append(newlist)
            else:
                config.LOGGER.info('Too few training updates from user timeline')
                self.svd = None
        except Exception as ex:
            config.LOGGER.error("Error %s computing SVD and kmeans from user history for mission %s", ex['msg'], self.missionId)

    def make_recommend(self, end_time=datetime.utcnow(), minutes_prior=15, top=10):

        working_size = top * 2
        working_list=[]
        result_list=[]
        try:
            config.LOGGER.info('Generation content recommendations for user %s', self.account['profile']['preferredUsername'])
            if self.svd is not None:
                tweets=self.get_input_updates(end_time, minutes=minutes_prior, maximum=1000)
                config.LOGGER.debug('%d updates from account timeline read from database', len(tweets))
                if len(tweets) < top:
                    return []

                tokenized_tweets = [ ' '.join(doc['keywords']) for doc in tweets]
                Y = self.vectorizer.transform(tokenized_tweets)
                svdY = self.svd.transform(Y)
                y_transform = self.k_means.transform(svdY)
                # terms = self.vectorizer.get_feature_names()

                all_cluster_dist = []
                y_predict=self.k_means.predict(svdY)

                for i in range(self.cluster_count):
                    cluster_distance=[]
                    for j in range(len(y_predict)):
                        if y_predict[j] == i and sum(svdY[j]) != 0.0:
                            cluster_distance.append({'index':j, 'cluster': i, 'dist':np.sqrt(sum([y*y for y in y_transform[j]]))})
                    newlist=sorted(cluster_distance, key=operator.itemgetter('dist'), reverse=False)
                    all_cluster_dist.append(newlist)
                # cluster_topics = self.topics(5)
                for tuple in map(None, *all_cluster_dist):
                    for entry in tuple:
                        if entry != None:
                            cluster_example=[]
                            cluster_idx = entry['cluster']
                            cluster_entry = self.all_cluster_dist[cluster_idx][0]
                            cluster_example.append(self.training_docs[cluster_entry['index']]['text'])
                            cluster_entry = self.all_cluster_dist[cluster_idx][1]
                            cluster_example.append(self.training_docs[cluster_entry['index']]['text'])
                            working_list.append({"dist":entry['dist'],"text": tweets[entry['index']]['text'],"id":str(tweets[entry['index']]['_id']), "sender":str(tweets[entry['index']]['sender']), 'samples':cluster_example})
                        if len(working_list)==working_size:
                            break
                    else:
                        continue # executed if the loop ended normally (no break)
                    break        # executed if 'continue' was skipped (break)


                result_list = sorted(working_list, key=operator.itemgetter('dist'), reverse=False)
            return result_list[:top]

        except Exception as ex:
            config.LOGGER.error("Error %s computing recommendations for mission %s", ex['msg'], self.missionId)
            return []
