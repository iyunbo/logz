"""create the pipeline
"""

from kedro.pipeline import Pipeline, node

from .nodes import train, predict


def create_pipeline():
    return Pipeline(
        [
            node(
                train,
                ["dataloader",
                 "params:model_dir",
                 "num_classes",
                 "params:window_size",
                 "params:batch_size",
                 "params:num_epochs",
                 "params:lstm_input_size",
                 "params:lstm_hidden_size",
                 "params:lstm_num_layers",
                 "params:device",
                 ],
                "model_file",
            ),
            node(
                predict,
                ["num_classes",
                 "model_file",
                 "normal_sample",
                 "abnormal_sample",
                 "params:window_size",
                 "params:lstm_input_size",
                 "params:lstm_hidden_size",
                 "params:lstm_num_layers",
                 "params:num_predict_candidates",
                 "params:device",
                 ],
                None,
            )
        ]
    )
