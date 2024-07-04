from environs import Env

env = Env()
env.read_env()

with env.prefixed("GRAPHOSE_"):
    BIG_ARCHITECTURES = env.bool("BIG_ARCHITECTURES", False)
    REPRODUCIBLE = env.bool("REPRODUCIBLE", True)
