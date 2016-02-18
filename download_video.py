"""
Project for CS344

A simple tool that can download YouTube videos for testing.

Usage: python download_video.py <id> <filename>

Downloaded videos are saved in the videos/ folder.
"""

from sys import argv, exit
from pytube import YouTube

if len(argv) < 2:
    print 'Usage: python %s <ID> <filename>' % argv[0]
    exit()
yt = YouTube('https://www.youtube.com/watch?v=' + argv[1])
yt.set_filename(argv[2])
video = yt.get('mp4', '360p')
print 'Downloading...'
video.download('videos/')
print 'Video saved as videos/%s.mp4' % video.filename
