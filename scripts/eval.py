import argparse

from src.utils import create_graph_dataset


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Evaluation for graph-based recommendation system.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam'
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='test.csv',
        help="the evaluation csv datafile inside the dataset folder (e.g. test.csv)"
    )

    # parser.add_argument(
    #     '--checkpoint_id',
    #     type=int,
    #     required=True,
    #     help="use which checkpoint(.ckpt) file to infer"
    # )

    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default="cuda:0",
        help="device id"
    )

    return parser.parse_args()


def evaluate():
    """Evaluation process."""
    args = get_args()

    data_dict = create_graph_dataset("data/steam.csv")

if __name__ == '__main__':
    evaluate()