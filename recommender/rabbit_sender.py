import config
import pika



amqp_url = 'amqp://guest:guest@localhost:5672/%2F'


class RabbitPublisher(object):
    """This is an example publisher that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, it will reopen it. You should
    look at the output, as there are limited reasons why the connection may
    be closed, which usually are tied to permission related issues or
    socket timeouts.

    It uses delivery confirmations and illustrates one way to keep track of
    messages that have been sent and if they've been confirmed by RabbitMQ.

    """

    def __init__(self, host='localhost', port=5672, user='guest', pwd='guest', exchange='plover', heartbeat=3600):
        config.LOGGER.info('Instantiating and initializing publisher object')
        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self.host = host
        self.port = port
        self.user = user
        self.pwd=pwd
        self.exchange = exchange
        self.heartbeat = heartbeat
        self._stopping = False
        self._deliveries = []
        self._acked = 0
        self._nacked = 0
        self._message_number = 0

    def publish(self, message, routing_key):
        config.LOGGER.info('Publishing message')
        config.LOGGER.debug('Message is %s', message)
        parameters=pika.ConnectionParameters(host=self.host,
                    port=self.port,
                    credentials=pika.PlainCredentials(self.user, self.pwd))

        config.LOGGER.info('Opening blocking connection')
        conn = pika.BlockingConnection(parameters)
        channel = conn.channel()

        #ch.exchange_declare(exchange=self.exchange_name, type="topic", durable=True, auto_delete=False)
        config.LOGGER.info('Opening publishing message')
        channel.basic_publish(exchange=self.exchange,
                         routing_key=routing_key,
                         body=message,
                         properties=pika.BasicProperties(
                                content_type = "text/plain",
                                delivery_mode = 2, # persistent
                                ))

        config.LOGGER.info('Closing channel')
        conn.close()


