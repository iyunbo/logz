"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team, Yunbo WANG
License     : MIT
"""

import hashlib
import logging
import os
import re
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)


class Logcluster:
    def __init__(self, log_template='', log_idl=None):
        self.log_template = log_template
        if log_idl is None:
            log_idl = []
        self.log_idl = log_idl


class Node:
    def __init__(self, child_node=None, depth=0, digit_or_token=None):
        if child_node is None:
            child_node = dict()
        self.child_node = child_node
        self.depth = depth
        self.digit_or_token = digit_or_token


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4,
                 max_child=100, rex=None, keep_param=True):
        """
        Attributes
        ----------
            log_format : the format expression of raw log messages
            indir : the input path stores the input log file name
            outdir : the output path stores the file containing structured logs
            depth : depth of all leaf nodes
            st : similarity threshold
            max_child : max number of children of an internal node
            rex : regular expressions used in preprocessing (step1)
            keep_param : indicates if we keep the parameters' values
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = max_child
        self.log_name = None
        self.save_path = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex if rex is not None else []
        self.keep_param = keep_param

    @staticmethod
    def has_numbers(s):
        return any(char.isdigit() for char in s)

    def tree_search(self, root_node, seq):
        ret_log_cluster = None

        seq_len = len(seq)
        if seq_len not in root_node.child_node:
            return ret_log_cluster

        parent_node = root_node.child_node[seq_len]

        current_depth = 1
        for token in seq:
            if current_depth >= self.depth or current_depth > seq_len:
                break

            if token in parent_node.child_node:
                parent_node = parent_node.child_node[token]
            elif '<*>' in parent_node.child_node:
                parent_node = parent_node.child_node['<*>']
            else:
                return ret_log_cluster
            current_depth += 1

        log_cluster_l = parent_node.child_node

        ret_log_cluster = self.fast_match(log_cluster_l, seq)

        return ret_log_cluster

    def add_seq_to_prefix_tree(self, rn, log_cluster):
        seq_len = len(log_cluster.log_template)
        if seq_len not in rn.child_node:
            first_layer_node = Node(depth=1, digit_or_token=seq_len)
            rn.child_node[seq_len] = first_layer_node
        else:
            first_layer_node = rn.child_node[seq_len]

        parent_node = first_layer_node

        current_depth = 1
        for token in log_cluster.log_template:

            # Add current log cluster to the leaf node
            if current_depth >= self.depth or current_depth > seq_len:
                if len(parent_node.child_node) == 0:
                    parent_node.child_node = [log_cluster]
                else:
                    parent_node.child_node.append(log_cluster)
                break

            # If token not matched in this layer of existing tree.
            if token not in parent_node.child_node:
                if not self.has_numbers(token):
                    if '<*>' in parent_node.child_node:
                        if len(parent_node.child_node) < self.maxChild:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parent_node.child_node[token] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.child_node['<*>']
                    else:
                        if len(parent_node.child_node) + 1 < self.maxChild:
                            new_node = Node(depth=current_depth + 1, digit_or_token=token)
                            parent_node.child_node[token] = new_node
                            parent_node = new_node
                        elif len(parent_node.child_node) + 1 == self.maxChild:
                            new_node = Node(depth=current_depth + 1, digit_or_token='<*>')
                            parent_node.child_node['<*>'] = new_node
                            parent_node = new_node
                        else:
                            parent_node = parent_node.child_node['<*>']

                else:
                    if '<*>' not in parent_node.child_node:
                        new_node = Node(depth=current_depth + 1, digit_or_token='<*>')
                        parent_node.child_node['<*>'] = new_node
                        parent_node = new_node
                    else:
                        parent_node = parent_node.child_node['<*>']

            # If the token is matched
            else:
                parent_node = parent_node.child_node[token]

            current_depth += 1

    # seq1 is template
    @staticmethod
    def seq_dist(template, sequence):
        assert len(template) == len(sequence)
        sim_tokens = 0
        num_of_param = 0

        for token1, token2 in zip(template, sequence):
            if token1 == '<*>':
                num_of_param += 1
                continue
            if token1 == token2:
                sim_tokens += 1

        ret_val = float(sim_tokens) / len(template)

        return ret_val, num_of_param

    def fast_match(self, log_cluster_l, seq):
        ret_log_cluster = None

        max_sim = -1
        max_num_param = -1
        max_cluster = None

        for log_cluster in log_cluster_l:
            current_similarity, current_num_param = self.seq_dist(log_cluster.log_template, seq)
            if current_similarity > max_sim or (current_similarity == max_sim and current_num_param > max_num_param):
                max_sim = current_similarity
                max_num_param = current_num_param
                max_cluster = log_cluster

        if max_sim >= self.st:
            ret_log_cluster = max_cluster

        return ret_log_cluster

    @staticmethod
    def get_template(seq1, seq2):
        assert len(seq1) == len(seq2)
        result = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                result.append(word)
            else:
                result.append('<*>')

            i += 1

        return result

    def output_result(self, log_cluster_l):
        log_templates = [0] * self.df_log.shape[0]
        log_template_ids = [0] * self.df_log.shape[0]
        df_events = []
        for log_cluster in log_cluster_l:
            template_str = ' '.join(log_cluster.log_template)
            occurrence = len(log_cluster.log_idl)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in log_cluster.log_idl:
                logID -= 1
                log_templates[logID] = template_str
                log_template_ids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        # df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_template_ids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_param:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.save_path, self.log_name + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.save_path, self.log_name + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"])

    def print_tree(self, node, dep):
        output = ''
        for i in range(dep):
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
            self.print_tree(node.child_node[child], dep + 1)

    def parse(self, log_name):
        log.info('Parsing file: ' + os.path.join(self.path, log_name))
        start_time = datetime.now()
        self.log_name = log_name
        root_node = Node()
        log_clu_l = []

        self.load_data()

        count = 0
        for idx, line in self.df_log.iterrows():
            log_id = line['LineId']
            message_template = self.preprocess(line['Content']).strip().split()
            # message_template = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            match_cluster = self.tree_search(root_node, message_template)

            # Match no existing log cluster
            if match_cluster is None:
                new_cluster = Logcluster(log_template=str(message_template), log_idl=[log_id])
                log_clu_l.append(new_cluster)
                self.add_seq_to_prefix_tree(root_node, new_cluster)

            # Add the new log message to the existing cluster
            else:
                new_template = self.get_template(message_template, match_cluster.logTemplate)
                match_cluster.logIDL.append(log_id)
                if ' '.join(new_template) != ' '.join(match_cluster.logTemplate):
                    match_cluster.logTemplate = new_template

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                log.info('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.output_result(log_clu_l)

        log.info('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_log_format_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.log_name), regex, headers)

    def preprocess(self, line):
        for current_rex in self.rex:
            line = re.sub(current_rex, '<*>', line)
        return line

    @staticmethod
    def log_to_dataframe(log_file, regex, headers):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        line_count = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                # noinspection PyBroadException
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    line_count += 1
                except Exception as e:
                    log.error("error", e)

        log_df = pd.DataFrame(log_messages, columns=headers)
        log_df.insert(0, 'LineId', None)
        log_df['LineId'] = [i + 1 for i in range(line_count)]
        return log_df

    @staticmethod
    def generate_log_format_regex(log_format):
        """ Function to generate regular expression to split log messages
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
    def get_parameter_list(row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("<*>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
