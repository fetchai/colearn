from colearn_examples_new.new_demo import main, TaskType
import argparse
import os

parser = argparse.ArgumentParser(description='Run colearn demo')
parser.add_argument("-d", "--train_dir", default=None, help="Directory for training data")
parser.add_argument("-e", "--test_dir", default=None, help="Directory for test data")

parser.add_argument("-t", "--task", default="PYTORCH_XRAY",
                    help="Options are " + " ".join(str(x.name)
                                                   for x in TaskType))

parser.add_argument("-m", "--model_type", default=None, type=str)

parser.add_argument("-n", "--n_learners", default=5, type=int)
parser.add_argument("-p", "--n_epochs", default=15, type=int)

parser.add_argument("-v", "--vote_threshold", default=0.5, type=float)

parser.add_argument("-r", "--train_ratio", default=0.8, type=float)

parser.add_argument("-s", "--seed", type=int, default=None)
parser.add_argument("-l", "--learning_rate", type=float, default=None)
parser.add_argument("-b", "--batch_size", type=int, default=None)

args = parser.parse_args()

# Generate seed
if args.seed is None:
    args.seed = int.from_bytes(os.urandom(4), byteorder="big")

# Print seed for logs
print("Seed is ", args.seed)

# Optional arguments - will be replaced by default values depending on task/model if not set
optional_learning_kwargs = dict()
if args.learning_rate is not None:
    optional_learning_kwargs["learning_rate"] = args.learning_rate
if args.batch_size is not None:
    optional_learning_kwargs["batch_size"] = args.batch_size

main(str_task_type=args.task,
     train_data_folder=args.train_dir,
     test_data_folder=args.test_dir,
     n_learners=args.n_learners,
     n_epochs=args.n_epochs,
     str_model_type=args.model_type,
     vote_threshold=args.vote_threshold,
     seed=args.seed,
     shuffle_seed=args.seed,
     **optional_learning_kwargs)
