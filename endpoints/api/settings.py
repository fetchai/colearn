import os

from api.schemas import Info

DEVELOPMENT_DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'colearn.db')
DATABASE_PATH = DEVELOPMENT_DATABASE_PATH
DEFAULT_PAGE_SIZE = 50

# this file will hold the static information about the node
node_info = Info(
    name='node-name',
    identity='',
    driver='none',
    version='v0',
)
