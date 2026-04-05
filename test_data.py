import importlib.util
import sys
sys.path.insert(0, "/home/sv6234/HSML/dynamic_hsml/HSML_Dynamic")

spec = importlib.util.spec_from_file_location(
    "hsml_main",
    "/home/sv6234/HSML/dynamic_hsml/HSML_Dynamic/main.py",
)
hsml_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hsml_main)  # registers flags
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

# Set flags needed by the generator
FLAGS.datasource = "mnist_permutations"
FLAGS.datadir = "/home/sv6234/HSML/La-MAML/data"
FLAGS.recurring_perm_mnist = True

from data_generator import MNIST_Permutations_DataGenerator

gen = MNIST_Permutations_DataGenerator(FLAGS.datadir)
print("num_total_tasks:", gen.num_total_tasks)
print("task_order:", gen.task_order)
print("recurring_to_orig (first 10):", list(gen.recurring_to_orig.items())[:10])
