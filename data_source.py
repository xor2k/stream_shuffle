from os import listdir
import numpy as np
import time
import ext.sequence_shuffle
import sys
import imageio
import cv2

data_dir = './data'

names_and_offsets_dtype = np.dtype([
    ('filename', 'U32'),
    ('begin_offset', np.uint64),
    ('end_offset', np.uint64),
    ('frame_count', np.uint64)])

labels_dtype = np.dtype([
    ('filename', 'U32'),
    ('ins', np.uint64),
    ('outs', np.uint64)]
)

class data_source():
    def __init__(self):
        self.plain_filename = data_dir+'/data.npy'
        self.names_and_offsets_filename = data_dir+'/index.npy'
        self.plain_data = np.array([], dtype=np.single)
        self.names_and_offsets = \
            np.array([], dtype=names_and_offsets_dtype)
        self.offsets_only = np.array([], dtype=np.uint64)

    def enable(self):
        self.plain_data = np.load(self.plain_filename, mmap_mode='r')
        self.names_and_offsets = np.load(self.names_and_offsets_filename)
        labels = np.sort(np.loadtxt(
            'data/labels.csv', skiprows=1, delimiter=',', dtype=labels_dtype,
            usecols=[1,2,3]
        ), order='filename')
        self.labels = np.empty(self.names_and_offsets.shape, np.dtype([
            ('ins', np.uint64),
            ('outs', np.uint64)])
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
        return self.plain_data[e['begin_offset']:e['end_offset']]

    def shuffle(self):
        # some placeholder labels
        labels_in = self.labels[:]['ins']
        labels_out = self.labels[:]['outs']
        batch_size = 16
        ext.sequence_shuffle.shuffle(
            batch_size,
            8,
            12,
            self.plain_data,
            self.names_and_offsets[:]['begin_offset'],
            labels_in,
            labels_out
        )
        t0 = time.time()
        for i in range(1, batch_size):
            ext.sequence_shuffle.create_batch()
            lower_in = ext.sequence_shuffle.get_y_lower_in()
            X = ext.sequence_shuffle.get_X()
            # print(lower_in.tolist())
            # print(lower_in)

            # vid_writer = imageio.get_writer('video.mp4', fps=20)
            # for j in range(1, 1000):
            #     # as_image = np.reshape(np.floor(X[j]*256), (25, 20, 1))
            #     as_image = np.reshape(np.floor(X[j]*256), (20, 25))
            #     scaled_image = cv2.resize(as_image.astype('uint8'), (32, 32))
            #     # print(X[j]*256)
            #     vid_writer.append_data(scaled_image)
            # vid_writer.close()

            # if i == 1:
            #     exit()

        t1 = time.time()
        print((t1-t0))
        # print(sequence_shuffle.shuffle(arr, 3.0))