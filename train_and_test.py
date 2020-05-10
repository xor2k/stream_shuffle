import util
import math
import numpy as np
import data_source
from pathlib import Path

train_part = 0.6
test_part = 1-train_part

class train_and_test:
    def __init__(self, data_source):
        self.train_filename = Path("train.csv")
        self.test_filename = Path("test.csv")
        self.data_source = data_source
        self.create_train_test_split(np.arange(
            data_source.names_and_offsets.size, dtype=util.index_dtype
        ))
    
    def load(self):
        names_and_offsets = self.data_source.names_and_offsets
        def indices_from_file(filename):
            return np.argwhere(np.isin(
                names_and_offsets['filename'],
                np.unique(np.sort(np.array(
                    filename.read_text().splitlines(),
                    names_and_offsets.dtype['filename']
                ))), assume_unique=True
            )).reshape(-1).astype(util.index_dtype)

        self.train_indices = indices_from_file(self.train_filename)
        self.test_indices = indices_from_file(self.test_filename)

    def create_train_test_split(self, shuffling):
        names_and_offsets = self.data_source.names_and_offsets
        train_test_barrier = math.floor(names_and_offsets.shape[0] * train_part)
        self.train_indices = np.sort(shuffling[:train_test_barrier])
        self.test_indices = np.sort(shuffling[train_test_barrier:])
    
    def shuffle(self):
        self.create_train_test_split(util.make_shuffling(
            self.data_source.names_and_offsets.size
        ))

    def create_train_file(self):
        self.train_filename.write_text('\n'.join(list(
            self.train_files["filename"]
        ))+'\n')
    
    def create_test_file(self):
        self.test_filename.write_text('\n'.join(list(
            self.test_files["filename"]
        ))+'\n')

if __name__ == "__main__":
    np.random.seed(0)
    
    ds = data_source.data_source()
    tt = train_and_test(ds)
    tt.shuffle()
    
    if not tt.train_filename.exists():
        tt.create_train_file()

    if not tt.test_filename.exists():
        tt.create_test_file()