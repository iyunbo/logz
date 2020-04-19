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
                 max_child=100, rex=None, keep_para=True):
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
            keep_para : indicates if we keep the parameters' values
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = max_child
        self.log_name = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex if rex is not None else []
        self.keep_para = keep_para

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

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.log_template)
        if seqLen not in rn.child_node:
            firtLayerNode = Node(depth=1, digit_or_token=seqLen)
            rn.child_node[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.child_node[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.log_template:

            # Add current log cluster to the leaf node
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.child_node) == 0:
                    parentn.child_node = [logClust]
                else:
                    parentn.child_node.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.child_node:
                if not self.has_numbers(token):
                    if '<*>' in parentn.child_node:
                        if len(parentn.child_node) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digit_or_token=token)
                            parentn.child_node[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.child_node['<*>']
                    else:
                        if len(parentn.child_node) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digit_or_token=token)
                            parentn.child_node[token] = newNode
                            parentn = newNode
                        elif len(parentn.child_node) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digit_or_token='<*>')
                            parentn.child_node['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.child_node['<*>']

                else:
                    if '<*>' not in parentn.child_node:
                        newNode = Node(depth=currentDepth + 1, digit_or_token='<*>')
                        parentn.child_node['<*>'] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.child_node['<*>']

            # If the token is matched
            else:
                parentn = parentn.child_node[token]

            currentDepth += 1

    # seq1 is template
    def seq_dist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

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

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            template_str = ' '.join(logClust.log_template)
            occurrence = len(logClust.log_idl)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.log_idl:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.log_name + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.log_name + '_templates.csv'), index=False,
                        columns=["EventId", "EventTemplate", "Occurrences"])

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        log.info(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.child_node:
            self.printTree(node.child_node[child], dep + 1)

    def parse(self, logName):
        log.info('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.log_name = logName
        rootNode = Node()
        logCluL = []

        self.load_data()

        count = 0
        for idx, line in self.df_log.iterrows():
            logID = line['LineId']
            message_template = self.preprocess(line['Content']).strip().split()
            # message_template = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            matchCluster = self.tree_search(rootNode, message_template)

            # Match no existing log cluster
            if matchCluster is None:
                newCluster = Logcluster(log_template=str(message_template), log_idl=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                newTemplate = self.get_template(message_template, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate

            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                log.info('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        self.outputResult(logCluL)

        log.info('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.log_name), regex, headers, self.log_format)

    def preprocess(self, line):
        for current_rex in self.rex:
            line = re.sub(current_rex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
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

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'\\s+', template_regex)
        template_regex = "^" + template_regex.replace("<*>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
