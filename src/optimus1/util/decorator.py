def retry(func, max_retry: int = 3):
    def wrapper(*args, **kwargs):
        retry = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retry += 1
                if retry > max_retry:
                    raise e

    return wrapper
