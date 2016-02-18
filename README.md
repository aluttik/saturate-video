# saturate-video

Uses a convolutional neural network to saturate black and white videos.

### Installation

First, make sure that you have both ffmpeg and tensorflow installed.  
From there, you'll need to torrent the [VGG16 Tensorflow model](http://tinyclouds.org/colorize/colorize-20160110.tgz.torrent).
Finally, run:
```
pip install -r requirements.txt
```

### Usage

You can download videos from youtube by running:
```
python download_video.py <id> <filename>
```

That will save videos in the `videos/` directory.

To colorize the videos, run:
```
python -W ignore saturate_video.py <filename> [fps]
```

### Example

To see an example of the output, check out 
https://www.youtube.com/watch?v=ntSA9C2zrx0.
