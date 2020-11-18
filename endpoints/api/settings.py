import os

from api.schemas import Info


def _build_database_path():
    database_root = os.environ.get('DATABASE_PATH', PROJECT_ROOT)
    return os.path.join(database_root, DATABASE_BASE_NAME)


DATABASE_BASE_NAME = 'colearn.db'
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATABASE_PATH = _build_database_path()
DEFAULT_PAGE_SIZE = 50

# this file will hold the static information about the node
node_info = Info(
    name='node-name',
    identity='',
    driver='none',
    version='v0',
)


def configure_node_info(name: str, identity: str, driver: str, version: str) -> None:
    """
    Called by the parent library to configure the APIs node information

    :param name: The name of the node
    :param identity: The public key identity of the
    :param driver: The type of the driver this API is attached to
    :param version: The version of the driver this API is attached to
    :return: None
    """
    global node_info
    node_info = Info(
        name=name,
        identity=identity,
        driver=driver,
        version=version,
    )
