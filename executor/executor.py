# Importing all needed libraries.
import json
from queue import PriorityQueue, Queue
import threading
import torch


class TaskExecutorManager:
    def __init__(self, config : "ConfigManager", word_embedder : "WordEmbeder") -> None:
        '''
            This function creates and sets up the Task Executor Manager.
            Task Executor Manager executes all tasks that come to the service.
                :param config: ConfigManager
                    The configuration manager.
                :parma word_embedder: WordEmbedder
                    THe Word Embedding object used to get the embeddings from text.
        '''
        # Setting up the neural network and word embedding dependencies.
        self.model = torch.load(config.model_path)
        self.model.eval()
        self.index2intent_mapper = json.load(open(config.index2intent_mapper_path, "r"))
        self.word_embedder = word_embedder

        # Setting up the concurrency dependencies.
        self.task_number_limit = config.task_number_limit
        self.active_task_number = 0
        self.priority_queue = PriorityQueue()
        self.stop_process_queue = Queue()
        self.stop_process_queue_lock = threading.Lock()
        self.priority_queue_lock = threading.Lock()
        self.task_number_limit_lock = threading.Lock()

        # Starting the prediction threads.
        for _ in range(self.task_number_limit):
            threading.Thread(target=self.execute).start()
        print("threads started")

    def available_process_num(self) -> int:
        '''
            This function returns the number of available threads for processing.
                :return: int
                    The number of available processes.
        '''
        # Acquiring the queue and task number limit Locks.
        self.priority_queue_lock.acquire()
        self.task_number_limit_lock.acquire()

        # Calculating the number of processes registered in the Task Executor.
        process_num = self.active_task_number + self.priority_queue.qsize()

        # Releasing the queue and task number limit Locks.
        self.task_number_limit_lock.release()
        self.priority_queue_lock.release()
        return self.task_number_limit - process_num

    def add_to_queue(self, task : "Task") -> None:
        '''
            This function adds a Task to the execution queue.
                :param task: Task
                    The task that is submitted to execution by the service.
        '''
        # Acquiring the task number limit lock and checking the availability for new task.
        self.task_number_limit_lock.acquire()
        if self.active_task_number + self.priority_queue.qsize() < self.task_number_limit:

            # Adding the task to the execution queue.
            self.priority_queue_lock.acquire()

            # Computing the compute lock time and queue waiting time.
            task.compute_lock_time()
            task.set_timer_queue_waiting_time()
            self.priority_queue.put((-task.arrival_time, task))
            self.priority_queue_lock.release()
        self.task_number_limit_lock.release()

    def increase(self) -> None:
        '''
            This function creates a new execution process.
        '''
        self.task_number_limit_lock.acquire()
        self.task_number_limit += 1
        self.task_number_limit_lock.release()
        threading.Thread(target=self.execute).start()

    def decrease(self):
        '''
            This function stops a execution process.
        '''
        self.task_number_limit_lock.acquire()
        self.stop_process_queue_lock.acquire()
        self.task_number_limit -=1
        # Putting the in stop process queue the "stop" message.
        self.stop_process_queue.put("stop")
        self.stop_process_queue_lock.release()
        self.task_number_limit_lock.release()

    def execute(self) -> None:
        '''
            This function executes tasks by prediction the Intent of the text
            in the task.
        '''
        while True:
            # Acquiring the stop process queue lock and checking if there are values.
            self.stop_process_queue_lock.acquire()
            if self.stop_process_queue.qsize() > 0:
                # Getting the message from the queue.
                msg = self.stop_process_queue.get()

                if msg == "stop":
                    # Stopping the process.
                    self.stop_process_queue_lock.release()
                    break
            self.stop_process_queue_lock.release()

            # Acquiring the execution queue lock and checking if there are any tasks in the queue.
            self.priority_queue_lock.acquire()
            if self.priority_queue.qsize() > 0:
                # Getting a task from the queue.
                task = self.priority_queue.get()[1]

                # Computing the task queue waiting time.
                task.compute_queue_waiting_time()
                self.priority_queue_lock.release()

                # Increasing the number of active tasks.
                self.task_number_limit_lock.acquire()
                self.active_task_number += 1

                # Setting the active tasks number metric.
                task.set_thread_capacity(self.active_task_number / self.task_number_limit)
                self.task_number_limit_lock.release()

                # Setting the waiting queue length metric.
                task.set_waiting_queue_length(
                    self.priority_queue.qsize()
                )

                # Prediction of the intent.
                task.set_timer_actual_processing()

                # Getting the embeddings of the text.
                embeds = self.word_embedder.get_vectors(task.text)

                # Predicting the intent.
                pred = self.index2intent_mapper[
                    str(self.model(torch.stack([embeds]))[0].argmax().item())
                ]
                task.prediction = pred

                # Computing the actual processing time.
                task.compute_actual_processing()

                # Decreasing the number of active tasks.
                self.task_number_limit_lock.acquire()
                self.active_task_number -= 1
                self.task_number_limit_lock.release()

                # Notifying the service about finished execution of the task.
                with task.condition:
                    task.notify()
            else:
                self.priority_queue_lock.release()
