"""
Project for CS344

This program uses a trained Convolutional Neural Network to saturate black and
white videos.

Usage: python -W ignore saturate_video.py <filename> [fps]

Saturated videos are saved in the results/ folder.

@author: Arie van Luttikhuizen
@version: Fall 2015
"""

from math import ceil, floor
from sys import argv, exit
from os.path import basename
from moviepy.decorators import use_clip_fps_by_default
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.fx.all import resize, margin, blackwhite
from multiprocessing import cpu_count
from skimage import img_as_float, img_as_ubyte
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class ColorInferrenceNetwork:

    def __init__(self):
        # parse graph definition from saved tensorflow model
        with open("vgg16.tfmodel", mode='rb') as model:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model.read())
        # import graph definition and grayscale placeholder into tensorflow
        self.grayscale = tf.placeholder("float", [1, 224, 224, 1])
        tf.import_graph_def(graph_def, input_map={"grayscale": self.grayscale})

    def saturate_frame(self, frame):
        # convert the numpy array to floats between 0 and 1
        frame = img_as_float(frame.astype(np.uint8))
        # desaturate and reshape array so it can be used as input
        frame = np.average(frame, 2).reshape(1, 224, 224, 1)
        # open tensorflow session
        with tf.Session() as sess:
            # run the saturate tensor on the grayscale frame
            saturate = sess.graph.get_tensor_by_name("import/inferred_rgb:0")
            return sess.run(saturate, feed_dict={self.grayscale: frame})[0]

    @use_clip_fps_by_default
    def saturate_clip(self, clip, fps=None):
        # extract frames from the clip and saturate them
        new_frames = []
        total = int(clip.duration * fps) + 1
        for frame in tqdm(clip.iter_frames(fps), total=total):
            frame = img_as_ubyte(self.saturate_frame(frame))
            new_frames.append(frame)
        # create and return a new clip from the saturated frames
        new_clip = ImageSequenceClip(new_frames, fps=fps)
        new_clip = new_clip.set_audio(clip.audio)
        return new_clip


def load_video(filename):
    # load clip from file name and resize it to 224px wide
    clip = VideoFileClip(filename).fx(resize, width=224)
    # desaturate the clip, center it vertically, then return it
    m = (224 - clip.h) / 2.0
    return clip.fx(blackwhite).fx(margin, top=ceil(m), bottom=floor(m))


def save_video(clip, filename='out.mp4'):
    fname = 'results/' + basename(filename)
    clip.write_videofile(fname, threads=cpu_count(), preset='slow',
                         codec='libx264', audio_codec='aac', verbose=False,
                         temp_audiofile='temp-audio.m4a', remove_temp=True)


if __name__ == '__main__':
    if len(argv) < 2:
        print 'Usage: python -W ignore %s <filename> [fps]' % argv[0]
        exit()
    cnn = ColorInferrenceNetwork()
    filename = argv[1]
    grayscale_clip = load_video(filename)
    fps = grayscale_clip.fps if len(argv) < 3 else float(argv[2])
    saturated_clip = cnn.saturate_clip(grayscale_clip, fps=fps)
    save_video(saturated_clip, filename)
