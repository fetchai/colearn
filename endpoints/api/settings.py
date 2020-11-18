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
