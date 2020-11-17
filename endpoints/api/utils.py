from typing import Optional, Any, Iterable

from api.settings import DEFAULT_PAGE_SIZE


def default(v: Optional[Any], d: Any) -> Any:
    if v is None:
        return d
    else:
        return v


def paginate(model_type: Any, values: Iterable[Any], page: Optional[int], page_size: Optional[int]) -> Any:
    value_list = list(values)

    page = default(page, 0)
    page_size = default(page_size, DEFAULT_PAGE_SIZE)
    if len(value_list) == 0:
        total_pages = 1
    else:
        total_pages = ((len(value_list) + (page_size - 1)) // page_size)

    start_index = page * page_size
    last_index = start_index + page_size

    page_info = {
        'current_page': page,
        'total_pages': total_pages,
        'is_start': page == 0,
        'is_last': (page + 1) == total_pages,
        'items': value_list[start_index:last_index]
    }

    return model_type(**page_info)
