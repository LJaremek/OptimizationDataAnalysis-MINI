import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Binary PyTorch MLP")
    parser.add_argument(
        "--model",
        type=str,
        choices=["bimlp", "mlp", "qmlp"],
        default="bi_mlp",
        help="model to use (default: binary mlp (bimlp))",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamax", "adadelta"],
        default="adadelta",
        help="optimizer to use (default: adadelta)",
    )
    parser.add_argument("--no-cuda", action="store_true", help="disables CUDA training")
    parser.add_argument(
        "--dry-run", action="store_true", help="quickly check a single pass"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", help="For Saving the current Model"
    )
    return parser.parse_args()
