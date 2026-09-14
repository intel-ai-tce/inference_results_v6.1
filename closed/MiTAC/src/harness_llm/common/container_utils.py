def remove_value_from_dict(dictionary: dict, value):
    return {
        k: v for k, v in dictionary.items() if v != value
    }

def remove_none_from_dict(dictionary: dict):
    return remove_value_from_dict(dictionary, None)

def print_dict(data_dict, indent=0):
    for key, value in data_dict.items():
        prefix = '  ' * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            # Print list items in one line, comma-separated
            items = ', '.join(map(str, value))
            print(f"{prefix}{key}: {items}")
        else:
            print(f"{prefix}{key}: {value}")
