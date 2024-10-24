import dataclasses
@dataclasses.dataclass
class config:
    max_rows: int = int(1e5)
    dowload_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    download_dir: str = "/Users/Kenneth/PycharmProjects/24WS-mmd-code-public/rec_sys/data/"
    unzipped_dir: str = download_dir + "ml-25m/"
    file_path: str = download_dir + "ml-25m/ratings.csv"