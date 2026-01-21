# pylint: disable=C0200
from scipy.ndimage import gaussian_filter1d  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
# requires ffmpeg installed on your system
from matplotlib.animation import FFMpegWriter
from datetime import datetime
import os
from utils import *
from pathlib import Path
import sys

# Project root = current working directory
PROJECT_ROOT = Path.cwd()

# Results directory
results_dir = PROJECT_ROOT / "results_tuning"
results_dir.mkdir(exist_ok=True)