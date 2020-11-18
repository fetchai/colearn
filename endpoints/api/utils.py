from typing import Optional, Any, Iterable
from unittest import TestCase

from api.settings import DEFAULT_PAGE_SIZE


def default(optional_value: Optional[Any], default_value: Any) -> Any:
    """
    Given an optional value extract either the value or return a default if None

    :param optional_value: The optional value
    :param default_value: The default value is the optional value is None
    :return: The extracted value
    """
    if optional_value is None:
        return default_value
    else:
        return optional_value


def compute_total_pages(num_entries: int, page_size: int):
    """
    Compute the total pages based on a number of entries and a page size

    :param num_entries: The number of entries
    :param page_size: The page size
    :return: The page index
    """
    if num_entries == 0:
        return 1
    else:
        return (num_entries + (page_size - 1)) // page_size


def paginate(model_type: Any, values: Iterable[Any], page: Optional[int], page_size: Optional[int]) -> Any:
    """
    Paginate a in memory iterable

    :param model_type: The pydantic model type
    :param values: The iterable set of values
    :param page: The page index
    :param page_size: The size of a page
    :return: An instance of the model_type
    """
    value_list = list(values)

    page = default(page, 0)
    page_size = default(page_size, DEFAULT_PAGE_SIZE)
    total_pages = compute_total_pages(len(value_list), page_size)

    start_index = page * page_size
    last_index = start_index + page_size

    page_info = {
        'current_page': page,
        'total_pages': total_pages,
        'is_first': page == 0,
        'is_last': (page + 1) == total_pages,
        'items': value_list[start_index:last_index]
    }

    return model_type(**page_info)


def paginate_db(query, model, converter, page: Optional[int], page_size: Optional[int]):
    """
    Query and paginate the results from database, converting and validating the results

    :param query: A peewee based database query
    :param model: The pydantic model to be returned
    :param converter: A converter callable for translating the query results into list items
    :param page: The page index
    :param page_size: The page size
    :return: An instance of the `model`
    """
    page_index = default(page, 0)
    page_size = default(page_size, DEFAULT_PAGE_SIZE)
    total_pages = compute_total_pages(query.count(), page_size)

    return model(
        items=list(map(converter, query.paginate(page_index + 1, page_size))),
        current_page=page_index,
        total_pages=total_pages,
        is_first=page_index == 0,
        is_last=(page_index + 1) == total_pages,
    )


def aggregate_conditions(conditions):
    """
    Aggregate all the conditions with AND statements

    :param conditions: The list of conditions to be aggregated together
    :return: A combined aggregated
    """
    agg = conditions[0]
    for condition in conditions[1:]:
        agg &= condition
    return agg


class BasicEndpointTest(TestCase):
    def assertIsPage(self, resp, page: int, total_pages: int):
        self.assertEqual(resp['current_page'], page)
        self.assertEqual(resp['total_pages'], total_pages)
        self.assertEqual(resp['is_first'], page == 0)
        self.assertEqual(resp['is_last'], (page + 1) == total_pages)
