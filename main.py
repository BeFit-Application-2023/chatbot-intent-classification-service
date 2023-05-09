# Importing the external libraries.
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from flask_migrate import Migrate
import threading
import requests
import uuid

# Importing the internal libraries.
from executor.executor import TaskExecutorManager
from executor.task import Task
from word_embedders.factory import WordEmbedderFactory
from cerber import SecurityManager
from schemas import IntentTextSchema
from config import ConfigManager

# Loading the configuration from the configuration file.
config = ConfigManager("config.ini")

# Creation of the Security Manager.
security_manager = SecurityManager(config.security.secret_key)

# Setting up the sqlalchemy database uri.
sqlalchemy_database_uri = f"postgresql://{config.database.username}:{config.database.password}@{config.database.host}/{config.database.table}"

# Creation of the intent schema.
intent_schema = IntentTextSchema()

# Setting up the Flask dependencies.
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["SQLALCHEMY_DATABASE_URI"] = sqlalchemy_database_uri
app.secret_key = config.security.secret_key

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Loading the word embedding.
glove = WordEmbedderFactory().get_word_embedding(config.word_embedding_dict)

# Creation of the Task Executor.
TASK_EXECUTOR = TaskExecutorManager(config.neural_network, glove)

# Defining the IntentModel Dadabase.
class IntentsModel(db.Model):
    # Setting up the table name.
    __tablename__ = 'intents'

    # Setting up the column names and data types.
    id = db.Column(db.String(64), primary_key=True)
    correlation_id = db.Column(db.String(64), unique=False)
    text = db.Column(db.Text, unique=False)
    prediction = db.Column(db.String(32), unique=False)

    def __init__(self, index : str, text : str, correlation_id : str, prediction : str):
        '''
            This function is the constructor of the IntentsModel.
                :param index: str
                    The index to insert the record in the table.
                :param text: str
                    The text of the message.
                :param correlation_id: str
                    The correlation id of the request.
                :param prediction: str
                    The predicted intent of the model.
        '''
        self.id = index
        self.text = text
        self.correlation_id = correlation_id
        self.prediction = prediction

    def __repr__(self):
        '''
            This functions shows the record in specified format.
        '''
        return f"<text id={self.id} text={self.text} prediction={self.prediction} correlation_id={self.correlation_id}>"


# Creation of the tables in the database.
with app.app_context():
    db.create_all()
    db.session.commit()

while True:
    sidecar_hmac = SecurityManager(config.service_sidecar.secret_key)._SecurityManager__encode_hmac(
        config.generate_info_for_service_discovery()
    )
    resp = requests.post(
        f"http://{config.service_sidecar.host}:{config.service_sidecar.port}/{config.service_sidecar.register_endpoint}",
        json = config.generate_info_for_service_discovery(),
        headers={"Token" : sidecar_hmac}
    )
    if resp.status_code == 200:
        break


@app.route("/intent", methods=["GET"])
def intent():
    '''
        This function triggers when the /intent endpoint is called.
    '''
    # Checking the access token.
    check_response = security_manager.check_request(request)
    if check_response != "OK":
        return check_response, check_response["code"]
    else:
        status_code = 200

        # Validation of the json.
        result, status_code = intent_schema.validate_json(request.json)
        if status_code != 200:
            # If the request body didn't passed the json validation a error is returned.
            return result, status_code
        else:
            # Checking the number of available processes.
            if TASK_EXECUTOR.available_process_num() > 0:
                # Creation of the task.
                task = Task(
                    result["text"],
                    threading.Condition()
                )

                # Setting the time checkpoint for lock time metric.
                task.set_timer_lock_time()

                # Adding the task to queue.
                TASK_EXECUTOR.add_to_queue(task)

                # Waiting for the task to process.
                with task.condition:
                    task.condition.wait()

                # Generating the universally unique identifier.
                index = str(uuid.uuid4())

                # Setting the time checkpoint for database response metric.
                task.set_timer_db_response_time()

                # Creating a new record of Intent.
                new_intent_record = IntentsModel(
                    index,
                    result["text"],
                    request.json["correlation_id"],
                    task.prediction
                )

                # Adding the record to the database.
                db.session.add(new_intent_record)
                try:
                    db.session.commit()
                except Exception as e:
                    error = {
                        "name" : e.__class__.__name__,
                        "cause" : e.__cause__.__repr__()
                    }
                    print(error)
                    # Calculating the database response time metric.
                    task.compute_db_response_time()

                    # Adding the database error.
                    task.add_db_error(error)

                    return task.json(), 500

                # Calculating the database response time metric.
                task.compute_db_response_time()

                return task.json(), status_code
            else:
                # Returning error if there are to many requests.
                return {
                    "error_code" : 429,
                    "message" : "To much requests"
                }, 429

@app.route("/increase", methods=["POST"])
def increase():
    '''
        This function is triggered then the /increase endpoint is called.
        It increases the number of running processes on the Task Executor.
    '''
    # Checking the access token.
    check_response = security_manager.check_request(request)
    if check_response != "OK":
        return check_response, check_response["code"]
    else:
        # Increasing the number of executor processes on the Task Executor.
        TASK_EXECUTOR.increase()

        return {
            "message" : "The number of running threads was increased",
            "code" : 200
        }, 200

@app.route("/decrease", methods=["POST"])
def decrease():
    '''
        This function is triggered then the /decrease endpoint is called.
        It decreases the number of running processes on the Task Executor.
    '''
    # Checking the access token.
    check_response = security_manager.check_request(request)
    if check_response != "OK":
        return check_response, check_response["code"]
    else:
        # Decreases the number of executor processes on the Task Executor.
        TASK_EXECUTOR.decrease()
        return {
                   "message" : "The number of running threads was decreased",
                   "code" : 200
               }, 200

# Running the main flask module.
if __name__ == "__main__":
    app.run(
        port=config.general.port,
        #port=6001,
        host="0.0.0.0"
        #host="0.0.0.0"
    )
