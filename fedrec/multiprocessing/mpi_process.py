from typing import Dict
from fedrec.utilities import registry
from fedrec.trainers.base_trainer import BaseTrainer
from fedrec.multiprocessing.jobber import Jobber
from mpi4py import MPI


@registry.load("multiprocessing", "MPI")
class MPIProcess:
    """
    Construct an MPI Process Manager for Trainers

    Attributes
    ----------
    trainer : BaseTrainer
        Trainer executing on the actor
    logger : logger
        Logger Object
    com_manager_config : dict
        Communication of config manager stored as dictionary
    """
    def __init__(self,
                 trainer: BaseTrainer,
                 logger,
                 com_manager_config: Dict) -> None:
        self.pool = MPI.COMM_WORLD
        self.rank = self.pool.Get_rank()
        self.num_processes = self.pool.Get_size()
        self.jobber = Jobber(trainer=trainer, logger=logger)
        self.process_comm_manager = registry.construct(
            "comm_manager", config_dict=com_manager_config)

    def run(self) -> None:
        """
        After calling the function, the Communication 
        Manager listens to the queue for messages, 
        executes the job request and publishes the results 
        in that order. It will stop listening after receiving
        job_request with JOB_TYPE "STOP" 
        """
        while True:
            job_request = self.process_comm_manager.receive_message()
            if job_request.JOB_TYPE == "STOP":
                return

            result = self.jobber.run(job_request)
            self.publish(result)

    def publish(self, job_result) -> None:
        """
        Publishes the result after executing the job request
        """
        self.process_comm_manager.send_message(job_result.result())

    def stop(self) -> None:
        self.process_comm_manager.stop()
