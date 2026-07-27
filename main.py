# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:07:28 2026

@author: mamon
"""

from pathlib import Path
from src.load_sparc_data import load_all_galaxies

folder = Path("data/raw_data/galaxies")

galaxies = load_all_galaxies(folder)

print(len(galaxies))
print(galaxies.keys())