# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:50:45 2026

@author: mamon
"""

from pathlib import Path
import pandas as pd
import glob


def load_all_galaxies(folder: Path) -> dict:

    """   
Load all SPARC galaxy rotation curve files from a folder.

Parameters
----------
folder : Path
    Path to the folder containing the SPARC *_rotmod.dat files.

Returns
-------
dict
    Dictionary mapping galaxy names to pandas DataFrames.
"""

    files = folder.glob("*_rotmod.dat")

    galaxies = {}

    for file in files:
        galaxy_name = file.stem.replace("_rotmod", "") #remove file ext and _rotmod 

        data = pd.read_csv(
            file,
            sep=r"\s+",
            comment="#",
            header=None
            )

        data.columns = [
            "Rad",
            "Vobs",
            "errV",
            "Vgas",
            "Vdisk",
            "Vbul",
            "SBdisk",
            "SBbul"
            ]

        galaxies[galaxy_name] = data

    return galaxies






""""
print(galaxies.keys())
print(Path.cwd())
print("Working directory:", Path.cwd())
print("Folder exists:", folder.exists())
print("Folder path:", folder.resolve())
print("\nEverything in folder:")
for f in folder.glob("*"):
    print(f.name)
"""