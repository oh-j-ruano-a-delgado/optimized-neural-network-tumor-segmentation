"""Dataset package: download, patient-wise splits, preprocessing, augmentation and DataLoaders.

Importing any submodule runs this file first, so it disables albumentations' online version check
(``NO_ALBUMENTATIONS_UPDATE``) before ``transforms`` imports the library. Without it, every import
issues a network request, including the re-imports performed by spawned DataLoader workers on
Windows. ``setdefault`` leaves an explicit user setting untouched.
"""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
