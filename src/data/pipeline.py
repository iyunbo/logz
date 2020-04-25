"""create the pipeline
"""

from kedro.pipeline import Pipeline, node

from .nodes import parse_log, make_dataloader


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                parse_log,
                ["params:input_dir",
                 "params:log_format",
                 "params:log_file",
                 "params:parsing_result_dir",
                 "params:parsing_similarity_threshold",
                 "params:parsing_tree_depth",
                 "params:parsing_regex"
                 ],
                "structured_csv",
            ),
            node(
                make_dataloader,
                ["structured_csv",
                 "params:event_key",
                 "params:sequence_key",
                 "params:window_size",
                 "params:batch_size"],
                ["dataloader", "num_classes"],
            )
        ]
    )
