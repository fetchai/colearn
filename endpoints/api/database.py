from contextvars import ContextVar

from peewee import CharField, TextField, IntegerField, DoubleField, Model, BlobField, ForeignKeyField, BooleanField, \
    SqliteDatabase, _ConnectionState, BigIntegerField

from api.settings import DATABASE_PATH

db_state_default = {"closed": None, "conn": None, "ctx": None, "transactions": None}
db_state = ContextVar("db_state", default=db_state_default.copy())


class PeeweeConnectionState(_ConnectionState):
    def __init__(self, **kwargs):
        super().__setattr__("_state", db_state)
        super().__init__(**kwargs)

    def __setattr__(self, name, value):
        self._state.get()[name] = value

    def __getattr__(self, name):
        return self._state.get()[name]


db = SqliteDatabase(DATABASE_PATH, check_same_thread=False)
db._state = PeeweeConnectionState()


class DBDataset(Model):
    name = CharField(primary_key=True)
    loader_name = CharField()
    loader_params = TextField()
    location = CharField()
    seed = IntegerField(null=True)  # range? [1, 100] ? :grin:
    train_size = DoubleField()
    validation_size = DoubleField(null=True)
    test_size = DoubleField()

    class Meta:
        table_name = 'datasets'
        database = db  # This model uses the "people.db" database.


class DBModel(Model):
    name = CharField(primary_key=True)
    model = CharField()
    parameters = TextField(null=True)
    weights = BlobField(null=True)

    class Meta:
        table_name = 'models'
        database = db  # This model uses the "people.db" database.


class DBExperiment(Model):
    name = CharField(primary_key=True)
    training_mode = CharField(default="collective")
    model = ForeignKeyField(DBModel)
    dataset = ForeignKeyField(DBDataset)
    seed = IntegerField(null=True)  # if not present then pick one [1, 100]?

    # smart contract information
    contract_address = CharField(null=True)
    parameters = TextField(null=True)

    # status information
    is_owner = BooleanField(default=False)
    state = CharField(default='')
    epoch = IntegerField(default=-1)

    # statistics information
    mean_epoch_time = DoubleField(null=True)
    mean_train_time = DoubleField(null=True)
    mean_evaluation_time = DoubleField(null=True)

    class Meta:
        table_name = 'experiments'
        database = db  # This model uses the "people.db" database.


class DBVote(Model):
    experiment = ForeignKeyField(DBExperiment)
    epoch = BigIntegerField(index=True)
    vote = BooleanField()
    is_proposer = BooleanField()

    class Meta:
        table_name = 'votes'
        database = db  # This model uses the "people.db" database.


class DBPerformance(Model):
    experiment = ForeignKeyField(DBExperiment)
    epoch = BigIntegerField(index=True)
    mode = CharField(index=True)
    performance = DoubleField()

    class Meta:
        table_name = 'performance'
        database = db  # This model uses the "people.db" database.


db.connect()
db.create_tables([DBModel, DBDataset, DBExperiment, DBVote, DBPerformance])
db.close()
