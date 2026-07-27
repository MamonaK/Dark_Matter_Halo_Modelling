# -*- coding: utf-8 -*-
"""


@author: mamon
"""

from pathlib import Path
from src.load_sparc_data import load_all_galaxies

folder = Path("data/raw_data/galaxies")

galaxies = load_all_galaxies(folder)

print(len(galaxies))
print(galaxies.keys())