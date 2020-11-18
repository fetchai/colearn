import unittest

from api.settings import configure_node_info, get_node_info


class SettingsTests(unittest.TestCase):
    def test_update_for_node_info(self):
        configure_node_info('foo-node', 'public-identity', 'test-driver', 'v1.0.0')

        node_info = get_node_info()
        self.assertEqual(node_info.name, 'foo-node')
        self.assertEqual(node_info.identity, 'public-identity')
        self.assertEqual(node_info.driver, 'test-driver')
        self.assertEqual(node_info.version, 'v1.0.0')
