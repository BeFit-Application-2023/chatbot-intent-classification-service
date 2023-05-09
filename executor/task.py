# Importing all needed libraries.
import time
import psutil


class Task:
    def __init__(self, text : str, condition : "threading.Condition") -> None:
        '''
            This class is and abstraction of the task executed by Task Execution Manager for
            keeping together all attributes of the task.
                :param text: str
                    The text on which is needed to make prediction.
                :param condition: threading.Condition
                    The threading Condition used to notify the service of the task execution end.
        '''
        self.text = text
        self.arrival_time = time.time()
        self.condition = condition
        self.prediction = None

        self.db_error = None

    def add_db_error(self, db_error_description : dict) -> None:
        '''
            This function adds to the task the error that appeared in the data base.
        '''
        self.db_error = db_error_description

    def set_timer_lock_time(self) -> None:
        '''
            This function sets the time checkpoint when the task started to wait for a lock
            to release.
        '''
        self.lock_time_per_process = time.time()

    def compute_lock_time(self) -> None:
        '''
            This function calculates the lock time per process.
        '''
        self.lock_time_per_process = time.time() - self.lock_time_per_process

    def set_timer_queue_waiting_time(self) -> None:
        '''
            This function sets the time checkpoint when the task was added to the queue.
        '''
        self.queue_waiting_time = time.time()

    def compute_queue_waiting_time(self) -> None:
        '''
            This function calculates the queue waiting time of the process.
        '''
        self.queue_waiting_time = time.time() - self.queue_waiting_time

    def set_timer_actual_processing(self) -> None:
        '''
            This function sets the time checkpoint when the prediction of the intent started.
        '''
        self.actual_processing = time.time()

    def compute_actual_processing(self) -> None:
        '''
            This function calculates the actual processing time of the task.
        '''
        self.actual_processing = time.time() - self.actual_processing

    def compute_task_service_time(self) -> None:
        '''
            This function calculates the task service time.
        '''
        self.task_service_time = time.time() - self.arrival_time

    def set_timer_db_response_time(self) -> None:
        '''
            This function sets up te time checkpoint of when the write to the database started.
        '''
        self.db_response_time = time.time()

    def compute_db_response_time(self) -> None:
        '''
            This function calculates the database response time of the task.
        '''
        self.db_response_time = time.time() - self.db_response_time

    def set_waiting_queue_length(self, queue_length) -> None:
        '''
            This function saves the queue length when the task is processed.
        '''
        self.queue_waiting_length = queue_length

    def set_thread_capacity(self, thread_capacity) -> None:
        '''
            This function saves the thread capacity when the task is processed.
        '''
        self.thread_capacity = thread_capacity

    def notify(self) -> None:
        '''
            This function notifies the service that the processing of the task has ended.
        '''
        self.condition.notify()

    def json(self) -> dict:
        '''
            This function converts the task into a dictionary.
        '''
        # Calculating the CPU utilization.
        cpu_utilization = psutil.cpu_percent(2)

        # Computing the task service time.
        self.compute_task_service_time()
        return {
            "text" : self.text,
            "prediction" : self.prediction,
            "latency" : {
                "lock_time" : self.lock_time_per_process,
                "queue_waiting_time" : self.queue_waiting_time,
                "actual_processing" : self.actual_processing,
                "task_service_time" : self.task_service_time,
                "database_response_time" : self.db_response_time
            },
            "saturation" : {
                "cpu_utilization" : cpu_utilization,
                "ram_utilization" : psutil.virtual_memory()[2],
                "waiting_queue_length" : self.queue_waiting_length,
                "thread_capacity" : self.thread_capacity
            },
            "errors" : {
                "db_error" : self.db_error
            }
        }