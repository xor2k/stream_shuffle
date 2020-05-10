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
        self.data_source = train_and_test.data_source
        self.train_and_test = train_and_test
        self.batch_size = 16
        self.aggregation_size = 8
        self.batch_video_count = self.batch_size * self.aggregation_size
        self.video_count = self.data_source.names_and_offsets.shape[0]
        self.train_batch_count = \
            self.train_and_test.train_indices.size // self.batch_video_count

        # TODO might be problematic: last batch is skipped
        # TODO make this math.ceil one day
        self.test_batch_count = \
            self.train_and_test.test_indices.size // self.batch_video_count

    def make_shuffling(self):
        return util.make_shuffling(self.train_and_test.train_indices.size)

    def train_batch(
        self, video_index_shuffling, batch_index, frame_step=1
    ):

        return self.make_batch(
            video_index_shuffling, batch_index, frame_step, DataMode.TRAIN_DATA
        )
    
    def test_batch(
        self, video_index_shuffling, batch_index, frame_step=1
    ):
        return self.make_batch(
            video_index_shuffling, batch_index, frame_step, DataMode.TEST_DATA
        )
    
    def make_batch(
        self, video_index_shuffling, batch_index, frame_step, data_mode
    ):
        ds = self.data_source
        tt = self.train_and_test
        batch_video_count = self.batch_video_count
        batch_size = self.batch_size
        aggregation_size = self.aggregation_size

        data = ds.data

        first_video_offset = batch_index*batch_video_count
        last_video_offset = (batch_index+1)*batch_video_count

        inner_loop_dimensions = (batch_size, aggregation_size)

        video_index_raw = tt.train_indices[
            video_index_shuffling[first_video_offset:last_video_offset]
        ] if data_mode == DataMode.TRAIN_DATA else tt.test_indices[
            first_video_offset:last_video_offset
        ]

        labels = np.reshape(ds.labels[video_index_raw], inner_loop_dimensions)

        video_frame_count = ds.names_and_offsets[video_index_raw]["frame_count"]

        data_video_offset = np.reshape(
            ds.names_and_offsets[video_index_raw]["begin_offset"],
            inner_loop_dimensions
        )

        # aggregation_video_filename = np.reshape(
        #     ds.names_and_offsets[video_index_raw]["filename"],
        #     inner_loop_dimensions
        # )

        aggregation_video_length = np.reshape(
            video_frame_count, inner_loop_dimensions
        )

        aggregation_video_offset = np.zeros_like(aggregation_video_length)
        aggregation_video_offset[:,1:] = np.cumsum(
            aggregation_video_length, axis=1
        )[:,0:-1]

        aggregation_frame_count = np.sum(aggregation_video_length, axis=1)

        aggregation_max_frame_count = np.amax(aggregation_frame_count)

        X_frames = np.empty(
            (aggregation_max_frame_count,batch_size)+data.shape[1:], dtype=data.dtype
        )

        y_boarding_lower_bound = np.zeros(
            (aggregation_max_frame_count, batch_size), dtype=data.dtype
        )
        y_alighting_lower_bound = np.zeros_like(y_boarding_lower_bound)

        y_boarding_upper_bound = np.zeros_like(y_boarding_lower_bound)
        y_alighting_upper_bound = np.zeros_like(y_boarding_lower_bound)

        for (batch_entry, aggregation_entry) in itertools.product(
            range(batch_size), range(aggregation_size)
        ):
            video_frame_count = aggregation_video_length[batch_entry][aggregation_entry]

            video_first_frame_index = data_video_offset[batch_entry][aggregation_entry]
            video_last_frame_index = video_first_frame_index + video_frame_count

            aggregation_first_frame = aggregation_video_offset[batch_entry][aggregation_entry]
            aggregation_last_frame = aggregation_first_frame + video_frame_count

            flip_vertical = util.randbit() if data_mode == DataMode.TRAIN_DATA else 0
            flip_temporal = util.randbit() if data_mode == DataMode.TRAIN_DATA else 0

            X_frames[
                aggregation_first_frame:aggregation_last_frame,batch_entry
            ] = data[video_first_frame_index:video_last_frame_index][
                ::(-1 if flip_temporal else 1)][:,::(-1 if flip_vertical else 1)
            ]

            boarding_entry = "alighting" if flip_temporal else "boarding"
            alighting_entry = "boarding" if flip_temporal else "alighting"

            boarding = labels[batch_entry][aggregation_entry][boarding_entry]
            alighting = labels[batch_entry][aggregation_entry][alighting_entry]

            door_close_frame = np.array(
                aggregation_last_frame-1, aggregation_last_frame.dtype
            )

            # if aggregation_last_frame != aggregation_frame_count[batch_entry]:
            y_boarding_lower_bound[door_close_frame, batch_entry] = boarding
            y_alighting_lower_bound[door_close_frame, batch_entry] = alighting

            y_boarding_upper_bound[aggregation_first_frame, batch_entry] = boarding
            y_alighting_upper_bound[aggregation_first_frame, batch_entry] = alighting

        y_boarding_lower_bound = np.cumsum(y_boarding_lower_bound, axis=0)
        y_boarding_upper_bound = np.cumsum(y_boarding_upper_bound, axis=0)
        y_alighting_lower_bound = np.cumsum(y_alighting_lower_bound, axis=0)
        y_alighting_upper_bound = np.cumsum(y_alighting_upper_bound, axis=0)

        return (
            aggregation_video_length,
            aggregation_frame_count,
            X_frames,
            y_boarding_lower_bound,
            y_boarding_upper_bound,
            y_alighting_lower_bound,
            y_alighting_upper_bound
        )