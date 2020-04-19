import logging
import os.path as path
import pathlib

from ..constant import PLACEHOLDER_PARAM
from ..data import drain
import helper

TEST_LOG = "test.log"

pwd = pathlib.Path(__file__).parent.absolute()
parser = drain.LogParser(log_format="<Date> <Time> <Level> <Component>: <Content>",
                         indir=path.join(pwd, "data"),
                         outdir=path.join(pwd, "data"))

helper.setup_log()
log = logging.getLogger(__name__)


def test_has_numbers():
    assert parser.has_numbers("xxx9xxx")
    assert parser.has_numbers("xxx9x1x")
    assert not parser.has_numbers("xxxaxxx")


def test_load_data():
    parser.log_name = TEST_LOG
    parser.load_data()
    df = parser.df_log
    logging.info(f"loaded df: {df.head()}")
    assert df.columns.size == 6, "should contains all components + ID"


def test_parsing_log():
    parser.parse(TEST_LOG)


def test_preprocess():
    line = "20/02/28 22:47:37 INFO DatabricksILoop$: Finished creating throwaway interpreter"
    processed = parser.preprocess(line)
    assert processed == line, "preprocessing should not change this line"


def test_get_template():
    seq1 = ["word1", "word2", "xxx", "yyy", "zzz"]
    seq2 = ["word1", "word2", "aaa", "yyy", "bbb"]
    result = parser.get_template(seq1, seq2)
    assert result == ["word1", "word2", "<*>", "yyy", "<*>"]


def test_seq_dist():
    line = ["20/02/28", "22:47:37", "INFO", "DatabricksILoop$", ":", "Finished creating throwaway interpreter"]
    template = [PLACEHOLDER_PARAM, PLACEHOLDER_PARAM, "INFO", PLACEHOLDER_PARAM, ":", PLACEHOLDER_PARAM]
    similarity, param_num = parser.seq_dist(template, line)
    assert similarity == 2.0 / 6
    assert param_num == 4
