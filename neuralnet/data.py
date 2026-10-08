

def split_dataset(data: list, n_workers: int, worker_id: int) -> list:
    chunk_size = len(data) // n_workers
    start = worker_id * chunk_size
    end = start + chunk_size
    return data[start:end]