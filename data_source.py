import numpy as np
import time
import ext.sequence_shuffle
import sys
from pathlib import Path

data_dir = Path('./data')

offsets_dtype = np.dtype(np.uint64)

data_dtype = np.single

filename_dtype = 'U32'

names_and_offsets_dtype = np.dtype([
    ('filename', filename_dtype),
    ('begin_offset', np.uint64),
    ('end_offset', np.uint64),
    ('frame_count', np.uint64)])

labels_dtype = np.dtype([
    ('filename', filename_dtype),
    ('boarding', np.uint64),
    ('alighting', np.uint64)]
)

plain_filename = data_dir / "data.npy"
names_and_offsets_filename = data_dir / "names_and_offsets.npy"
labels_filename = data_dir / 'labels.csv'

class data_source():
    def __init__(self):
        self.data = np.load(plain_filename, mmap_mode='r')
        self.names_and_offsets = np.load(names_and_offsets_filename)
        labels = np.sort(np.loadtxt(
            labels_filename, skiprows=1, delimiter=',', dtype=labels_dtype,
            usecols=[1,2,3]
        ), order='filename')
        self.labels = np.empty(self.names_and_offsets.shape, np.dtype([
            ('boarding', np.uint64),
            ('alighting', np.uint64)])
        )
        for name_and_offset in enumerate(self.names_and_offsets):
            filename = name_and_offset[1][0][:-4]
            tmp = np.array([(filename, 0, 0)], \
                dtype=labels_dtype
            )
            index = np.searchsorted(labels, tmp)[0]
            if index >= len(labels) or labels[index][0] != filename:
                sys.exit('no labels found for file '+filename)
            tmp2 = labels[index]
            self.labels[name_and_offset[0]] = (tmp2[1], tmp2[2])

    def list_files(self):
        return self.names_and_offsets[:]['filename']

    def get_file(self, filename):
        tmp = np.array([(filename, 0, 0, 0)], \
            dtype=names_and_offsets_dtype
        )
        index = np.searchsorted(self.names_and_offsets, tmp)[0]
        if index >= len(self.names_and_offsets):
            return None
        e = self.names_and_offsets[index]
        if e['filename'] != filename:
            return None
        return self.data[e['begin_offset']:e['end_offset']]

    def shuffle_c(self):
        # some placeholder labels
        labels_boarding = self.labels[:]['boarding']
        labels_alighting = self.labels[:]['alighting']
        batch_size = 16
        ext.sequence_shuffle.create_shuffle(
            batch_size,
            8,
            self.data.reshape((-1, 500)),
            self.names_and_offsets[:]['begin_offset']*500,
            labels_boarding,
            labels_alighting
        )
        batch_count = ext.sequence_shuffle.get_batch_count()
        t0 = time.time()
        for i in range(0, batch_count):
            print(i)
            ext.sequence_shuffle.create_batch()
            # TODO do something here

        t1 = time.time()
        print((t1-t0))
        # print(sequence_shuffle.shuffle(arr, 3.0))

if __name__ == "__main__":
    ds = data_source()
    frames = ds.get_file("001_20160526_030141.uff")
    data_source.create_video(frames, "video.mp4")