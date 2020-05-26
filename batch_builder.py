import numpy as np
import util
import itertools
from train_and_test import train_and_test
from data_source import data_source
from enum import Enum

class DataMode(Enum):
    TRAIN_DATA = 1
    TEST_DATA = 2

class batch_builder():
    def __init__(self, train_and_test):
        ds = train_and_test.data_source
        self.data_source = ds
        self.train_and_test = train_and_test
        self.batch_size = 16
        self.max_videos_per_batch_element = 6
        frame_counts = ds.names_and_offsets[train_and_test.train_indices]["frame_count"]
        self.max_frame_count = np.inf; # int(np.mean(frame_counts)+1) * self.max_videos_per_batch_element
        self.epochial = True
        self.completed_epochs_count = None
        self.next_video_index = None
        self.batch_video_count = self.batch_size * self.max_videos_per_batch_element
        self.train_index_permutation_iterator = None
        self.train_index_permutation = None
        self.video_count = self.data_source.names_and_offsets.shape[0]
        self.train_batch_count = \
            self.train_and_test.train_indices.size // self.batch_video_count

        # TODO might be problematic: last batch is skipped
        # TODO make this math.ceil one day
        self.test_batch_count = \
            self.train_and_test.test_indices.size // self.batch_video_count

        self.reset_stats()

    def reset_stats(self):
        self.used_frame_count = 0
        self.total_frame_count = 0

    def make_shuffling(self):
        return util.make_shuffling(self.train_and_test.train_indices.size)

    def train_batch(
        self, video_index_shuffling, batch_index, frame_step=1
    ):

        return self.make_batch(
            video_index_shuffling, batch_index, frame_step, DataMode.TRAIN_DATA
        )

    def test_batch(
        self, batch_index, frame_step=1
    ):
        return self.make_batch(
            range(self.test_batch_count), batch_index, frame_step, DataMode.TEST_DATA
        )

    def get_next_video_index(self):
        tt = self.train_and_test

        if self.epochial:

            if self.train_index_permutation is None or \
                self.train_index_permutation_iterator == tt.train_indices.size:

                self.train_index_permutation = np.random.permutation(tt.train_indices)
                self.train_index_permutation_iterator = 0

                if self.completed_epochs_count is None:
                    self.completed_epochs_count = 0
                else:
                    self.completed_epochs_count += 1

            retval = self.train_index_permutation[self.train_index_permutation_iterator] if \
                self.next_video_index is None else self.next_video_index

            self.train_index_permutation_iterator += 1

            return retval

        else:

            return np.random.choice(
                tt.train_indices
            ) if self.next_video_index == None else self.next_video_index

    def set_next_video_index(self, video_index):
        self.next_video_index = video_index

    def make_batch(self):
        ds = self.data_source
        tt = self.train_and_test
        batch_video_count = self.batch_video_count
        batch_size = self.batch_size
        max_frame_count = self.max_frame_count
        data = ds.data

        layout = []

        batch_element = 0

        while batch_element < batch_size:
            begin_offset = 0
            layout_batch_element = []
            for i in range(self.max_videos_per_batch_element):
                video_index = self.get_next_video_index()

                frame_count = ds.names_and_offsets[video_index]["frame_count"]
                end_offset = int(begin_offset + frame_count)
                data_first_frame = ds.names_and_offsets[video_index]["begin_offset"]
                data_last_frame = int(data_first_frame + frame_count)

                if i > 0 and end_offset > max_frame_count or \
                    i == self.max_videos_per_batch_element-1:

                    self.set_next_video_index(video_index)

                    break

                self.set_next_video_index(None)

                flip_vertical = np.random.choice([0,1])
                flip_temporal = np.random.choice([0,1])

                boarding_entry = "alighting" if flip_temporal else "boarding"
                alighting_entry = "boarding" if flip_temporal else "alighting"

                boarding = ds.labels[video_index][boarding_entry]
                alighting = ds.labels[video_index][alighting_entry]

                layout_batch_element.append((
                    begin_offset, end_offset, data_first_frame, data_last_frame,
                    flip_vertical, flip_temporal, boarding, alighting
                ))

                begin_offset = end_offset

            batch_element += 1
            layout.append(layout_batch_element)

        max_frame_count = max(e[-1][1] for e in layout)
        self.total_frame_count += max_frame_count*batch_size
        self.used_frame_count += sum(e[-1][1] for e in layout)

        x = np.empty(
            (max_frame_count, batch_size) + data.shape[1:], dtype=data.dtype
        )

        y = np.zeros((max_frame_count, batch_size, 8), dtype=data.dtype)

        for batch_element, tmp in enumerate(layout):
            end_offset = None

            for layout_batch_element in tmp:
                (
                    begin_offset, end_offset, data_first_frame, data_last_frame,
                    flip_vertical, flip_temporal, boarding, alighting
                ) = layout_batch_element

                door_close_frame = int(end_offset-1)

                x[
                    begin_offset:end_offset,batch_element
                ] = data[data_first_frame:data_last_frame][
                    ::(-1 if flip_temporal else 1)
                ][
                    :,:,::(-1 if flip_vertical else 1)
                ]

                y[door_close_frame, batch_element, (0,1,6,7)] = \
                    [boarding, alighting, 1, 1]

                y[begin_offset, batch_element, 2:4] = [boarding, alighting]

            y[:end_offset,batch_element,4:6] = 1

        for i in range(4):
            y[:,:,i] = np.cumsum(y[:,:,i], axis=0)

        return x, y