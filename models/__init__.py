# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from .reltr import build


def build_model(args):
    return build(args)
