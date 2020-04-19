"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import hashlib
import logging
import os
import re
from datetime import datetime
from re import Pattern
from typing import Union, List

import pandas as pd

from ..constant import PLACEHOLDER_PARAM, EMPTY_TOKEN

log = logging.getLogger(__name__)


class LogCluster:
    def __init__(self, sequence=Union[list, None], id_list=Union[list, None]):
        """
        A log cluster maintains a token sequence with all matching identifiers

        :param sequence: whole token sequence
        :param id_list: list containing all matching identifiers
        """
        self.sequence = sequence
        self.id_list = id_list if id_list is not None else []


class Node:
    def __init__(self, child_node=None, depth=0, digit_or_token=None):
        """
        A Node represents a general sub-tree with a root node and a list of children.
        The children number is not fixed and we store the children list into a dict.
        The first layer of the tree stores the dict in form of {sequence length : sub-tree}

        :param child_node: children list
        :param depth: the depth of this subtree
        :param digit_or_token: the content of the current node, it can be a token or the length of sequence
        """
        self.child_node = child_node if child_node is not None else dict()
        self.depth = depth
        self.digit_or_token = digit_or_token


class LogParser:
    def __init__(self, log_format: str, indir='./', outdir='./result/', depth=4, similarity_threshold=0.4,
                 max_child=100, preprocess_regex=None, keep_param=True):
        """
        This class implements the main algorithm of Drain parse

        :param log_format: the log format expression, eg: <Date> <Time> <Level> <Subject>: <Content>
        :param indir : the input path stores the input log file name
        :param outdir : the output path stores the file containing structured logs
        :param depth : depth of all leaf nodes
        :param similarity_threshold : similarity threshold
        :param max_child : max number of children of an internal node
        :param preprocess_regex : regular expressions used in preprocessing (step1)
        :param keep_param: indicate if keeping parameter values during the process
        """
        self.input_dir = indir
        self.depth = depth - 2
        self.similarity_threshold = similarity_threshold
        self.max_child = max_child
        self.log_name = None
        self.output_dir = outdir
        self.df_log = None
        self.log_format = log_format
        self.preprocess_regex = preprocess_regex if preprocess_regex is not None else []
        self.keep_param = keep_param

    @staticmethod
    def has_number(s: str):
        return any(char.isdigit() for char in s)

    def tree_search(self, root_node: Node, seq: List[str]) -> Union[LogCluster, None]:
        """
        Search the matching Log cluster in the tree.

        :param root_node: the root of the tree
        :param seq: the sequence of log tokens
        :return: matching LogCluster or None if none matches
        """
        target_cluster = None

        seq_len = len(seq)
        if seq_len not in root_node.child_node:
            return target_cluster

        parent_node = root_node.child_node[seq_len]

        current_depth = 1
        for token in seq:
            if current_depth >= self.depth or current_depth > seq_len:
                break

            if token in parent_node.child_node:
                parent_node = parent_node.child_node[token]
            elif PLACEHOLDER_PARAM in parent_node.child_node:
                parent_node = parent_node.child_node[PLACEHOLDER_PARAM]
            else:
                return target_cluster
            current_depth += 1

        cluster_list = parent_node.child_node

        target_cluster = self.fast_match(cluster_list, seq)

        return target_cluster

    def add_to_tree(self, root_node: Node, cluster: LogCluster):
        """
        Core algorithm for inserting a log cluster into existing token tree
        :param root_node: the root of the tree
        :param cluster: log cluster for insertion
        """
        seq_len = len(cluster.sequence)
        if seq_len not in root_node.child_node:
            first_layer_node = Node(depth=1, digit_or_token=seq_len)
            root_node.child_node[seq_len] = first_layer_node
        else:
            first_layer_node = root_node.child_node[seq_len]

        parent_node = first_layer_node

        current_depth = 1
        for token in cluster.sequence:

            # Add current log cluster to the leaf node
            if current_depth >= self.depth or current_depth > seq_len:
                if len(parent_node.child_node) == 0:
                    parent_node.child_node = [cluster]
                else:
                    parent_node.child_node.append(cluster)
                break

            # If token not matched in this layer of existing tree.
            if token not in parent_node.child_node:
                if not self.has_number(token):
                    if PLACEHOLDER_PARAM in parent_node.child_node:
                        if len(parent_node.child_node) < self.max_child:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parent_node.child_node[token] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.child_node[PLACEHOLDER_PARAM]
                    else:
                        if len(parent_node.child_node) + 1 < self.max_child:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parent_node.child_node[token] = new_node
                            parent_node = new_node
                        elif len(parent_node.child_node) + 1 == self.max_child:
                            new_node = Node(depth=current_depth + 1, digit_or_token=PLACEHOLDER_PARAM)
                            parent_node.child_node[PLACEHOLDER_PARAM] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.child_node[PLACEHOLDER_PARAM]

                else:
                    if PLACEHOLDER_PARAM not in parent_node.child_node:
                        new_node = Node(depth=current_depth + 1, digit_or_token=PLACEHOLDER_PARAM)
                        parent_node.child_node[PLACEHOLDER_PARAM] = new_node
                        parent_node = new_node
                    else:
                        parent_node = parent_node.child_node[PLACEHOLDER_PARAM]

            # If the token is matched
            else:
                parent_node = parent_node.child_node[token]

            current_depth += 1

    @staticmethod
    def seq_similarity(template: List[str], seq: List[str]) -> (float, int):
        """
        Calculate sequence similarity according to specified template

        :param template: the log template to consider
        :param seq: the sequence to analyse
        :return: the similarity between the sequence and the template, the number of detected parameters
        """
        assert len(template) == len(seq)
        similar_tokens = 0
        param_num = 0

        for token1, token2 in zip(template, seq):
            if token1 == PLACEHOLDER_PARAM:
                param_num += 1
                continue
            if token1 == token2:
                similar_tokens += 1

        similarity = float(similar_tokens) / len(template)

        return similarity, param_num

    def fast_match(self, cluster_list: List[LogCluster], seq: List[str]):
        """
        Find a match of sequence from a log cluster (see LogCluster) list

        :param cluster_list: log cluster list
        :param seq: candidate token sequence
        :return: the matched log cluster
        """
        ret_cluster = None

        max_sim = -1
        max_param_num = -1
        max_cluster = None

        for cluster in cluster_list:
            cur_sim, cur_param_num = self.seq_similarity(cluster.sequence, seq)
            if cur_sim > max_sim or (cur_sim == max_sim and cur_param_num > max_param_num):
                max_sim = cur_sim
                max_param_num = cur_param_num
                max_cluster = cluster

        if max_sim >= self.similarity_threshold:
            ret_cluster = max_cluster

        return ret_cluster

    @staticmethod
    def get_template(seq1: List[str], seq2: List[str]) -> List[str]:
        assert len(seq1) == len(seq2)
        result = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                result.append(word)
            else:
                result.append(PLACEHOLDER_PARAM)

            i += 1

        return result

    def output_result(self, cluster_list: List[LogCluster]):
        log_templates = [EMPTY_TOKEN] * self.df_log.shape[0]
        log_template_ids = [0] * self.df_log.shape[0]
        df_events = []
        for log_cluster in cluster_list:
            template_str = ' '.join(log_cluster.sequence)
            occurrence = len(log_cluster.id_list)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for log_id in log_cluster.id_list:
                log_id -= 1
                log_templates[log_id] = template_str
                log_template_ids[log_id] = template_id
            df_events.append([template_id, template_str, occurrence])

        self.df_log['EventId'] = log_template_ids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_param:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameters, axis=1)
        self.df_log.to_csv(os.path.join(self.output_dir, self.log_name + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.output_dir, self.log_name + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"])

    def print_tree(self, node: Node, depth: int):
        output = ''
        for i in range(depth):
            output += '\t'

        if node.depth == 0:
            output += 'Root'
        elif node.depth == 1:
            output += '<' + str(node.digit_or_token) + '>'
        else:
            output += node.digit_or_token

        log.info(output)

        if node.depth == self.depth:
            return 1
        for child in node.child_node:
            self.print_tree(node.child_node[child], depth + 1)

    def parse(self, log_name: str):
        """
        This is the main entry point for Drain log parser.
        The result will be write to output_dir

        :param log_name: the file name of the log
        """
        print('Parsing file: ' + os.path.join(self.input_dir, log_name))
        log.info('Parsing file: ' + os.path.join(self.input_dir, log_name))
        start_time = datetime.now()
        self.log_name = log_name
        root_node = Node()
        cluster_list = []

        self.load_data()

        count = 0
        for idx, line in self.df_log.iterrows():
            log_id = line['LineId']
            message_seq = self.preprocess(line['Content']).strip().split()
            match_cluster = self.tree_search(root_node, message_seq)

            # Match no existing log cluster
            if match_cluster is None:
                new_cluster = LogCluster(sequence=message_seq, id_list=[log_id])
                cluster_list.append(new_cluster)
                self.add_to_tree(root_node, new_cluster)

            # Add the new log message to the existing cluster
            else:
                new_template = self.get_template(message_seq, match_cluster.sequence)
                match_cluster.id_list.append(log_id)
                if ' '.join(new_template) != ' '.join(match_cluster.sequence):
                    match_cluster.sequence = new_template

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))
                log.info('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_result(cluster_list)

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        log.info('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_format_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.input_dir, self.log_name), regex, headers)

    def preprocess(self, line: str) -> str:
        for current_rex in self.preprocess_regex:
            line = re.sub(current_rex, PLACEHOLDER_PARAM, line)
        return line

    @staticmethod
    def log_to_dataframe(log_file, regex, headers):
        """
        Function to transform log file to dataframe

        :param log_file: the log file name
        :param regex: the regex template representing the log file format
        :param headers: columns of structured log
        :return: pandas dataframe
        """
        log_messages = []
        line_count = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    line_count += 1

        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df.insert(0, 'LineId', None)
        log_df['LineId'] = [i + 1 for i in range(line_count)]
        return log_df

    @staticmethod
    def generate_format_regex(log_format: str) -> (List[str], Pattern):
        """ Function to generate regular expression to split log messages

        :param log_format: log format expression string, eg: <xx> <yy>: <zz>
        :return: the regex for splitting the log message
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', log_format)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', r'\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    @staticmethod
    def get_parameters(row: pd.Series) -> List[str]:
        """
        Extract parameter values from a data row

        :param row: data row
        :return: a list of parameter values according to its template
        """
        template_regex = re.sub(r"<[^<>]+>", PLACEHOLDER_PARAM, row["EventTemplate"])
        if PLACEHOLDER_PARAM not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        escaped_param_placeholder = re.sub(r'([^A-Za-z0-9])', r'\\\1', PLACEHOLDER_PARAM)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace(escaped_param_placeholder, "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
