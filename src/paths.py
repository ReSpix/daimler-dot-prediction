from pathlib import Path

ROOT = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()

TRAIN_PATH = ROOT / "data" / "daimler_mixtures_train.csv"
TEST_PATH = ROOT / "data" / "daimler_mixtures_test.csv"
PROPS_PATH = ROOT / "data" / "daimler_component_properties.csv"
MODEL_PATH = ROOT / "weights" / "best_model.pt"