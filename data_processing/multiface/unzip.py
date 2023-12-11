import os
import glob 
import argparse
import json
import logging
import multiprocessing as mp
import os

import requests
from bs4 import BeautifulSoup

MAX_TRY = 50
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Dataset Download")


def worker(path):
    if path[-4:] == ".tar":
        try_count = 0
        while True and (try_count < MAX_TRY):
            try_count = try_count + 1
            os.system(
                "tar -xvf %s -C %s && touch %s.unzip"
                % (path, "/".join(path.split("/")[:-1]), path)
            )
            if os.path.isfile("%s.unzip" % (path)):
                os.system("rm " + path)
                break
            else:
                logging.info("Unzipped %s failed. Re-unzipping..." % (path))

        logging.info("Done %s" % (path))


def unzip(unzip_tar):
    """create pool to extract all"""
    pool = mp.Pool(min(mp.cpu_count(), len(unzip_tar)))  # number of workers
    pool.map(worker, unzip_tar)
    pool.close()

def main():
    # unzip tar
    tar_files = glob.glob("./*.tar")
    excluded_tar_files = ['./6795937audio.tar', './6795937images--E057_Cheeks_Puffed.tar']
    unzip_tar = [f for f in tar_files if f not in excluded_tar_files]
    unzip(unzip_tar)
if __name__ == "__main__":
    main()