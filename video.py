import imageio
import cv2

def create_video(frames, target_filename, fps=10):
    vid_writer = imageio.get_writer(target_filename, fps=fps)
    for j in range(0, frames.shape[0]):
        as_image = np.floor(frames[j]*256)
        scaled_image = cv2.resize(as_image.astype('uint8'), (256, 208))
        vid_writer.append_data(scaled_image)
    vid_writer.close()

if __name__ == "__main__":
    from data_source import data_source
    from batch_builder import batch_builder
    from train_and_test import train_and_test
    import numpy as np
    
    np.random.seed(0)

    ds = data_source()
    tt = train_and_test(ds)
    bb = batch_builder(tt)
    
    shuffling = bb.make_shuffling()

    (
        aggregation_video_length,
        aggregation_frame_count,
        X_frames,
        y_boarding_lower_bound,
        y_boarding_upper_bound,
        y_alighting_lower_bound,
        y_alighting_upper_bound
    ) = bb.train_batch(shuffling, 0)

    index = 0

    frames = X_frames[:,index]
    target_filename = "video.mp4"
    fps = 20

    vid_writer = imageio.get_writer(target_filename, fps=fps)
    for j in range(0, frames.shape[0]):
        as_image = np.floor(frames[j]*256)
        scaled_image = cv2.resize(as_image.astype('uint8'), (256, 208))

        def put_text(text, pos):
            cv2.putText(
                scaled_image, text, pos, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = 0.5,
                color = (255,255,255),
                lineType = 1
            )

        put_text("%d"%(y_boarding_lower_bound[:,index][j]), (10,20))
        put_text("%d"%(y_boarding_upper_bound[:,index][j]), (10,40))
        put_text("%d"%(y_alighting_lower_bound[:,index][j]), (226,20))
        put_text("%d"%(y_alighting_upper_bound[:,index][j]), (226,40))

        vid_writer.append_data(scaled_image)
    vid_writer.close()