import asyncio
import zmq
from zmq.asyncio import Context
from time import sleep
from abc import ABC, abstractmethod
from fedrec.utilities import registry
from global_comm_stream import CommunicationStream


class AbstractComManager(ABC):

    @abstractmethod
    def send_message(self):
        pass

    @abstractmethod
    def receive_message(self):
        pass

    @abstractmethod
    def close(self):
        pass
@registry.load("communications", "ZeroMQ")
class ZeroMQ(AbstractComManager):
    def __init__(self, is_subscriber=False):
        self.context = Context.instance()
        if is_subscriber:            
            self.subscriber = self.context.socket(zmq.SUB)
        else:
            self.publisher = self.context.socket(zmq.PUB)
        if self.publisher:
            print("Connecting to Port . . . . ./n")
            self.publisher.connect('tcp://127.0.0.1:2000')
        if self.subscriber:
            print('Connecting to port . . . . ./n')
            self.subscriber.bind('tcp://127.0.0.1:2000')
            self.subscriber.subscribe(b'')

        async def receive_message(self):
            return await self.subscriber.recv_multipart()        
        
        def send_message(message):
            print("Sending Message . . . . . /n")
            self.publisher.send_pyobj(message)

        def close(self):
            if self.publisher:
                self.publisher.close()
            elif self.subscriber:
                self.subscriber.close()
            else:
                self.context.term()
            

            

  

  


