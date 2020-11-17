from api.schemas import Info

DEFAULT_PAGE_SIZE = 50

# this file will hold the static information about the node
node_info = Info(
    name='node-name',
    identity='',
    driver='none',
    version='v0',
)
