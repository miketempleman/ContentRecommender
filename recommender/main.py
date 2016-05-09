

from rabbit_consumer import RabbitConsumer
import config

from rabbit_sender import RabbitPublisher
import json
from recommend import ContentRecommend
from dateutil import parser as dateparser
from datetime import datetime, timedelta
from bson import json_util
import resource

import argparse

# LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
#               '-35s %(lineno) -5d: %(message)s')
# LOGGER = logging.getLogger(__name__)


#{"recommend":"CONTENT", "example-quality":0.15, "example-count-min":2, "max":5, "missions":["5444a80ae4b01939bbf69f3c","5345ce62744e47ff0b4ed89a", "54449b97e4b08d74189d0afe", "544499dae4b08d74189d003b"]}

#{"recommend":"REBUILD", "missions":["5345ce62744e47ff0b4ed89a", "54449b97e4b08d74189d0afe", "544499dae4b08d74189d003b"]}
recommenders = {}

def content_recommend(message_object):

    if 'end' in message_object:
        end_time = dateparser.parse(message_object['end'])
    else:
        end_time = datetime.utcnow()
    example_quality = message_object['example-quality'] if 'example-quality' in message_object else 0.1
    example_count_min = message_object['example-count-min'] if 'example-count-min' in message_object else 1

    for mission_id in message_object['missions']:
        config.LOGGER.debug("Generating %d content recommendation for mission %s", message_object['max'], mission_id)
        try:
            check_recommender(mission_id)
            results=recommenders[mission_id].recommend_from_timeline(top=message_object['max'], end_time=end_time, quality=example_quality, min_examples=example_count_min)

            json_recommend = json.dumps({'recommender':'CONTENT','missionId':mission_id, 'end_time':end_time, 'posts':results}, default=json_util.default)
            config.LOGGER.debug(' %d content recommendations for mission %s were generated', len(results), mission_id)
            if len(results) > 0:
                config.LOGGER.debug('Sending recommendation to rabbit now:\n  %s', json_recommend)
                publisher.publish(message=json_recommend, routing_key='recommendation.#')

        except Exception as ex:
             config.LOGGER.error("Error %s getting recommendation for mission %s", ex.message, mission_id)

def check_recommender(mission_id):

    if mission_id in recommenders:
        expiration = datetime.utcnow()-timedelta(hours=args.expiration)
        if recommenders[mission_id].create_date < expiration:
            config.LOGGER.debug("Removing expired recommender for mission %s from cache", mission_id)
            rebuild_recommender(mission_id)
    else:
        rebuild_recommender(mission_id)

def rebuild_recommender(mission_id):
    try:
        if mission_id in recommenders:
            config.LOGGER.debug("Removing recommender for mission %s from cache", mission_id)
            del recommenders[mission_id]
        recommenders[mission_id] = ContentRecommend(mission_id=mission_id, db_name=args.dbname, db_port=args.dbport, db_host=args.dbhost)
        config.LOGGER.info("Created recommender cache for mission %s", mission_id)

    except Exception as ex:
        config.LOGGER.error("Error %s deleting/creating recommender cache for mission %s", ex.message, mission_id)



def recommend_service(body):
    global recommenders
    global args
    try:
        config.LOGGER.info("Servicing recommender request")
        message_object = json.loads(body)
        if message_object['recommend'] == 'CONTENT':
            config.LOGGER.info("Servicing recommender CONTENT request")
            content_recommend(message_object)
        elif message_object['recommend'] == 'REBUILD':
            config.LOGGER.info("Rebuilding all missions")
            for mission_id in message_object['missions']:
                rebuild_recommender(mission_id)
        else:
            config.LOGGER.error("%s is invalid recommend command ", message_object['recommend'])

        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        maxmem = args.maxmem
        config.LOGGER.debug("Memory used is %d", mem)
        if mem > maxmem or args.nocache:
            keys = recommenders.keys()
            for key in keys:
                val = recommenders.pop(key)
                del val


    except Exception as ex:
       config.LOGGER.error("Exception %s processing message body %s", ex.message, body)

class Args:
    pass

def main():

    global recommenders
    global publisher
    global args


    try:
        config.LOGGER.info("Beginning recommender loop")
        consumer = RabbitConsumer(queue='recommend', exchange='plover', client=recommend_service, routing_key='recommend.#', host=args.rabbitserver, user=args.rabbituser, pwd=args.rabbitpwd)
        publisher = RabbitPublisher(exchange='plover', host=args.rabbitserver, user=args.rabbituser, pwd=args.rabbitpwd)

        config.LOGGER.debug("Memory current level %d", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        consumer.run()
    except KeyboardInterrupt:
        consumer.stop()

    else:
        config.LOGGER.error("Exception in running recommender")



if __name__ == '__main__':
    args = Args()

    parser = argparse.ArgumentParser(description='Meshfire Recomender Server')
    parser.add_argument('-dbn','--dbname',  default='plover_development', help='Set the name of the database to use')
    parser.add_argument('-dh', '--dbhost', default='127.0.0.1', help='Set the hostname')
    parser.add_argument('-p', '--dbport', default=27017, help='Set the db port')

    parser.add_argument('-rs', '--rabbitserver', default='localhost', help='RabbitMQ Server name')
    parser.add_argument('-ru', '--rabbituser', default='guest', help='RabbitMQ user name')
    parser.add_argument('-pwd', '--rabbitpwd', default='guest', help='RabbitMQ password')
    parser.add_argument('-ll', '--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')
    parser.add_argument('-log', '--logfile', default=None, help='Set log file')
    parser.add_argument('-thd', '--threads', default=1, help='Number of processes to spawn. Default=1')
    parser.add_argument('-exp', '--expiration', default=12, help='Hours until the content training expires')
    parser.add_argument('-noauto', '--noauto', default=False, help='Hours until the content training expires')
    parser.add_argument('-maxmem','--maxmem', type=int, default=200, help='Maximum memory size (in MB) before flushing cache')
    parser.add_argument('-nocache','--nocache', default=False, help='Do not employ a cache to save training')

    parser.parse_args(namespace=args)
    config.init_logging(log_level=args.loglevel, logfile=args.logfile)

    print 'Starting Meshfire Content Recommender'
    print ' courtesy of scikit learn'
    print ' options:'
    print '  database host: %s' % args.dbhost
    print '  database port: %d' % args.dbport
    print '  database name: %s' % args.dbname
    print '  rabbit host: %s' % args.rabbitserver
    print '  rabbit user: %s' % args.rabbituser
    print '  rabbit pwd: *********'
    print '  recommender training expiration time: %d hours' % args.expiration
    print '  logging level: %s' % args.loglevel
    print '  logging file: %s' % args.logfile
    print '  threads: %d' % args.threads
    print '  Max memory (MB) allowed: %d' % args.maxmem
    args.maxmem = args.maxmem * 2**20
    print '  Cache disabled: %s' % args.nocache

    main()