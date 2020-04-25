import logging
import os.path as path
import pathlib

import pandas as pd

from . import helper
from ..data.drain import PLACEHOLDER_PARAM, SUFFIX_PARSED_LOG
from ..data import drain
from ..data import nodes

TEST_LOG = "test.log"

pwd = pathlib.Path(__file__).parent.absolute()
parser = drain.LogParser(log_format="<Date> <Time> <Level> <Component>: <Content>",
                         indir=path.join(pwd, "data"),
                         outdir=path.join(pwd, "data"))

helper.setup_log()
log = logging.getLogger(__name__)


def test_has_numbers():
    assert parser.has_number("xxx9xxx")
    assert parser.has_number("xxx9x1x")
    assert not parser.has_number("xxxaxxx")


def test_load_data():
    parser.log_name = TEST_LOG
    parser.load_data()
    df = parser.df_log
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
    similarity, param_num = parser.seq_similarity(template, line)
    assert similarity == 2.0 / 6
    assert param_num == 4


def test_fast_match():
    line = ["20/02/28", "22:47:37", "INFO", "DatabricksILoop$", ":", "Finished creating throwaway interpreter"]
    template1 = [PLACEHOLDER_PARAM, PLACEHOLDER_PARAM, "INFO", "DatabricksILoop$", ":", PLACEHOLDER_PARAM]
    template2 = [PLACEHOLDER_PARAM, PLACEHOLDER_PARAM, "DEBUG", "DatabricksILoop$", ":", PLACEHOLDER_PARAM]
    cluster1 = drain.LogCluster(template1)
    cluster2 = drain.LogCluster(template2)
    matched_cluster = parser.fast_match([cluster1, cluster2], line)
    assert matched_cluster == cluster1


def test_match_with_greater_param_number():
    line = ["20/02/28", "22:47:37", "INFO", "DatabricksILoop$", ":", "Finished creating throwaway interpreter"]
    template1 = ["20/02/28", PLACEHOLDER_PARAM, "DEBUG", "DatabricksILoop$", ":", PLACEHOLDER_PARAM]
    template2 = [PLACEHOLDER_PARAM, PLACEHOLDER_PARAM, "INFO", "DatabricksILoop$", ":", PLACEHOLDER_PARAM]
    cluster1 = drain.LogCluster(template1)
    cluster2 = drain.LogCluster(template2)
    matched_cluster = parser.fast_match([cluster1, cluster2], line)
    assert matched_cluster == cluster2


def test_tree_search():
    root = drain.Node()
    cluster1 = drain.LogCluster(sequence=["xyz", PLACEHOLDER_PARAM], id_list=["1"])
    cluster2 = drain.LogCluster(sequence=[PLACEHOLDER_PARAM, "aaa"], id_list=["2"])
    cluster3 = drain.LogCluster(sequence=[PLACEHOLDER_PARAM, "kkk"], id_list=["3"])
    parser.add_to_tree(root, cluster1)
    parser.add_to_tree(root, cluster2)
    parser.add_to_tree(root, cluster3)
    match_cluster = parser.tree_search(root, ["xxx", "kkk"])
    assert match_cluster == cluster3, f"[{PLACEHOLDER_PARAM}, kkk] should match [xxx, kkk]"


def test_tree_search_should_return_none_if_length_miss_match():
    line = ["20/02/28", "22:47:37", "INFO", "DatabricksILoop$", ":", "Finished creating throwaway interpreter"]
    tree = drain.Node({1: drain.Node(), 2: drain.Node()}, depth=3, digit_or_token="xxx")
    cluster = parser.tree_search(tree, line)
    assert cluster is None


def test_add_to_tree():
    cluster1 = drain.LogCluster(sequence=["xxx", "yyy"], id_list=["1"])
    cluster2 = drain.LogCluster(sequence=["zzz", "aaa"], id_list=["2"])
    root = drain.Node()
    parser.add_to_tree(root, cluster1)
    parser.add_to_tree(root, cluster2)
    assert root.depth == 0, "root node should have depth 0"
    assert root.child_node[2].depth == 1, "first layer (size of 2) should have depth 1"
    assert len(root.child_node[2].child_node) == 2, "we added two node, so the second layer should have 2 children"
    assert root.child_node[2].child_node["xxx"].depth == 2, "the second layer (xxx) should have depth 2"
    assert root.child_node[2].child_node["zzz"].depth == 2, "the second layer (zzz) should have depth 2"
    assert len(root.child_node[2].child_node["xxx"].child_node) == 1, "node(xxx) only has one child"
    assert len(root.child_node[2].child_node["zzz"].child_node) == 1, "node(zzz) only has one child"
    assert root.child_node[2].child_node["xxx"].child_node[0].sequence == ["xxx", "yyy"], \
        "node(xxx) should contain (xxx, yyy)"
    assert root.child_node[2].child_node["zzz"].child_node[0].sequence == ["zzz", "aaa"], \
        "node(xxx) should contain (zzz, aaa)"


def test_print_tree():
    root = drain.Node()
    parser.add_to_tree(root, drain.LogCluster(sequence=["fff", "<*>"], id_list=["2"]))
    parser.add_to_tree(root, drain.LogCluster(sequence=["zzz", "ccc", "bbb"], id_list=["3"]))
    parser.add_to_tree(root, drain.LogCluster(sequence=["yyy", "ddd", "bbb", "eee"], id_list=["4"]))
    parser.print_tree(root, 0)


def test_get_parameter_list():
    row = pd.Series({"Content": "20/02/28 22:47:36 INFO JettyClient$: xxx",
                     "Component": "JettyClient$",
                     "EventTemplate": "<Date> <Time> <Level> <Component>: <Content>"})
    params = parser.get_parameters(row)

    assert len(params) == 5


def test_make_dataloader():
    file = nodes.parse_log(input_dir=path.join(pwd, "data"), log_format="<Date> <Time> <Level> <Component>: <Content>",
                           log_file=TEST_LOG, result_dir=path.join(pwd, "data"), similarity_threshold=0.3, depth=3,
                           regex=None)
    assert file.endswith(SUFFIX_PARSED_LOG)

    dataloader, num_classes = nodes.make_dataloader(csv_file=file, event_key="EventTemplate", sequence_key="Event",
                                                    window_size=5, batch_size=2)
    assert num_classes == 24
    assert len(dataloader) == 16


def test_regex_generation():
    headers, regex = parser.generate_format_regex("<Date> <Time> <Number> <Component> <Content>")
    assert headers == ["Date", "Time", "Number", "Component", "Content"]
    match = regex.match("yyyyMMdd hh:mm:ss 120 thread some logging and bla bla")
    assert match
    assert match.group("Date") == "yyyyMMdd"
    assert match.group("Time") == "hh:mm:ss"
    assert match.group("Number") == "120"
    assert match.group("Component") == "thread"
    assert match.group("Content") == "some logging and bla bla"
