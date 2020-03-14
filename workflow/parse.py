"""step: parsing the log"""
import os.path as path

import mlflow

import workflow.config as config
from logparser import Drain


def run():
    """parsing log"""
    # Regular expression list for optional preprocessing (default: [])
    regex = []
    similarity_threshold = 0.6  # Similarity threshold
    depth = 4  # Depth of all leaf nodes

    # parsing log into structured CSV
    parser = Drain.LogParser(config.LOG_FORMAT, indir=config.INPUT_DIR,
                             outdir=config.OUTPUT_DIR, depth=depth, st=similarity_threshold, rex=regex)
    parser.parse(config.LOG_FILE)

    mlflow.log_artifacts(config.OUTPUT_DIR, "parsed_log")

    return path.join(config.OUTPUT_DIR, config.LOG_FILE + '_structured.csv')
