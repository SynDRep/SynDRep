# -*- coding: utf-8 -*-
"""get the model config"""


def get_config(model_name: str) -> dict:
    """
    Get the model configuration based on the given model name.

    :param model_name: the name of the model (e.g., "TransE", "DistMult", etc.)

    :return: a dictionary containing the model configuration
    """
    if model_name == "TransE":
        config = {
            "optuna": {},
            "pipeline": {
                "model": "TransE",
                "model_kwargs_ranges": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 6,
                        "high": 9,
                        "scale": "power_two",
                    },
                    "scoring_fct_norm": {"type": "int", "low": 1, "high": 2},
                },
                "training_loop": "slcwa",
                "training_loop_kwargs": {"automatic_memory_optimization": True},
                "optimizer": "sgd",
                "optimizer_kwargs": {"weight_decay": 0.0},
                "optimizer_kwargs_ranges": {
                    "lr": {"type": "float", "low": 0.0001, "high": 1.0, "scale": "log"}
                },
                "loss": "MarginRankingLoss",
                "loss_kwargs": {},
                "loss_kwargs_ranges": {
                    "margin": {"type": "float", "low": 0.5, "high": 10, "q": 1.0}
                },
                "regularizer": "NoRegularizer",
                "regularizer_kwargs": {},
                "regularizer_kwargs_ranges": {},
                "negative_sampler": "BasicNegativeSampler",
                "negative_sampler_kwargs": {},
                "negative_sampler_kwargs_ranges": {
                    "num_negs_per_pos": {"type": "int", "low": 1, "high": 50, "q": 1}
                },
                "evaluator": "RankBasedEvaluator",
                "evaluator_kwargs": {
                    "filtered": True,
                    "automatic_memory_optimization": True,
                },
                "evaluation_kwargs": {"batch_size": None},
                "training_kwargs": {"num_epochs": 1000, "label_smoothing": 0.0},
                "training_kwargs_ranges": {
                    "batch_size": {
                        "type": "int",
                        "low": 8,
                        "high": 11,
                        "scale": "power_two",
                    }
                },
                "stopper": "early",
                "stopper_kwargs": {
                    "frequency": 25,
                    "patience": 2,
                    "relative_delta": 0.002,
                },
                "n_trials": 100,
                "timeout": 86400,
                "metric": "hits@10",
                "direction": "maximize",
                "sampler": "random",
                "pruner": "nop",
            },
        }
    elif model_name == "TransR":
        config = {
            "optuna": {},
            "pipeline": {
                "model": "TransR",
                "model_kwargs_ranges": {
                    "embedding_dim": {"high": 256, "low": 16, "q": 16, "type": "int"},
                    "relation_dim": {"high": 256, "low": 16, "q": 16, "type": "int"},
                    "scoring_fct_norm": {"type": "int", "low": 1, "high": 2},
                },
                "training_loop": "slcwa",
                "training_loop_kwargs": {"automatic_memory_optimization": True},
                "optimizer": "sgd",
                "optimizer_kwargs": {"weight_decay": 0.0},
                "optimizer_kwargs_ranges": {
                    "lr": {"type": "float", "low": 0.0001, "high": 1.0, "scale": "log"}
                },
                "loss": "MarginRankingLoss",
                "loss_kwargs": {},
                "loss_kwargs_ranges": {
                    "margin": {"type": "float", "low": 0.5, "high": 10, "q": 1.0}
                },
                "negative_sampler": "BasicNegativeSampler",
                "negative_sampler_kwargs": {},
                "negative_sampler_kwargs_ranges": {
                    "num_negs_per_pos": {"type": "int", "low": 1, "high": 50, "q": 1}
                },
                "evaluator": "RankBasedEvaluator",
                "evaluator_kwargs": {
                    "filtered": True,
                    "automatic_memory_optimization": True,
                },
                "evaluation_kwargs": {"batch_size": None},
                "training_kwargs": {"num_epochs": 1000, "label_smoothing": 0.0},
                "training_kwargs_ranges": {
                    "batch_size": {
                        "type": "int",
                        "low": 8,
                        "high": 11,
                        "scale": "power_two",
                    }
                },
                "stopper": "early",
                "stopper_kwargs": {
                    "frequency": 25,
                    "patience": 2,
                    "relative_delta": 0.002,
                },
                "n_trials": 100,
                "timeout": 86400,
                "metric": "hits@10",
                "direction": "maximize",
                "sampler": "random",
                "pruner": "nop",
            },
        }
    elif model_name == "RotatE":
        config = {
            "optuna": {},
            "pipeline": {
                "model": "RotatE",
                "model_kwargs_ranges": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 6,
                        "high": 9,
                        "scale": "power_two",
                    }
                },
                "training_loop": "slcwa",
                "training_loop_kwargs": {"automatic_memory_optimization": True},
                "optimizer": "adam",
                "optimizer_kwargs": {"weight_decay": 0.0},
                "optimizer_kwargs_ranges": {
                    "lr": {"type": "float", "low": 0.0001, "high": 1.0, "scale": "log"}
                },
                "loss": "NSSALoss",
                "loss_kwargs": {},
                "loss_kwargs_ranges": {
                    "margin": {"type": "float", "low": 1, "high": 30, "q": 2.0},
                    "adversarial_temperature": {
                        "type": "float",
                        "low": 0.1,
                        "high": 1.0,
                        "q": 0.1,
                    },
                },
                "regularizer": "NoRegularizer",
                "regularizer_kwargs": {},
                "regularizer_kwargs_ranges": {},
                "negative_sampler": "BasicNegativeSampler",
                "negative_sampler_kwargs": {},
                "negative_sampler_kwargs_ranges": {
                    "num_negs_per_pos": {"type": "int", "low": 1, "high": 50, "q": 1}
                },
                "evaluator": "RankBasedEvaluator",
                "evaluator_kwargs": {
                    "filtered": True,
                    "automatic_memory_optimization": True,
                },
                "evaluation_kwargs": {"batch_size": None},
                "training_kwargs": {"num_epochs": 1000, "label_smoothing": 0.0},
                "training_kwargs_ranges": {
                    "batch_size": {
                        "type": "int",
                        "low": 8,
                        "high": 11,
                        "scale": "power_two",
                    }
                },
                "stopper": "early",
                "stopper_kwargs": {
                    "frequency": 25,
                    "patience": 4,
                    "relative_delta": 0.002,
                },
                "n_trials": 100,
                "timeout": 129600,
                "metric": "hits@10",
                "direction": "maximize",
                "sampler": "random",
                "pruner": "nop",
            },
        }

    elif model_name == "HolE":
        config = {
            "optuna": {},
            "pipeline": {
                "model": "HolE",
                "model_kwargs_ranges": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 6,
                        "high": 9,
                        "scale": "power_two",
                    }
                },
                "training_loop": "slcwa",
                "training_loop_kwargs": {"automatic_memory_optimization": True},
                "optimizer": "adagrad",
                "optimizer_kwargs": {"weight_decay": 0.0},
                "optimizer_kwargs_ranges": {
                    "lr": {"type": "float", "low": 0.0001, "high": 1.0, "scale": "log"}
                },
                "loss": "MarginRankingLoss",
                "loss_kwargs": {},
                "loss_kwargs_ranges": {
                    "margin": {"type": "float", "low": 0.5, "high": 10, "q": 1.0}
                },
                "negative_sampler": "BasicNegativeSampler",
                "negative_sampler_kwargs": {},
                "negative_sampler_kwargs_ranges": {
                    "num_negs_per_pos": {"type": "int", "low": 1, "high": 50, "q": 1}
                },
                "evaluator": "RankBasedEvaluator",
                "evaluator_kwargs": {
                    "filtered": True,
                    "automatic_memory_optimization": True,
                },
                "evaluation_kwargs": {"batch_size": None},
                "training_kwargs": {"num_epochs": 1000, "label_smoothing": 0.0},
                "training_kwargs_ranges": {
                    "batch_size": {
                        "type": "int",
                        "low": 8,
                        "high": 11,
                        "scale": "power_two",
                    }
                },
                "stopper": "early",
                "stopper_kwargs": {
                    "frequency": 25,
                    "patience": 2,
                    "relative_delta": 0.002,
                },
                "n_trials": 100,
                "timeout": 129600,
                "metric": "hits@10",
                "direction": "maximize",
                "sampler": "random",
                "pruner": "nop",
            },
        }

    elif model_name == "ComplEx":
        config = {
            "optuna": {},
            "pipeline": {
                "model": "ComplEx",
                "model_kwargs_ranges": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 6,
                        "high": 9,
                        "scale": "power_two",
                    }
                },
                "training_loop": "slcwa",
                "training_loop_kwargs": {"automatic_memory_optimization": True},
                "optimizer": "adagrad",
                "optimizer_kwargs": {"weight_decay": 0.0},
                "optimizer_kwargs_ranges": {
                    "lr": {"type": "float", "low": 0.0001, "high": 1.0, "scale": "log"}
                },
                "loss": "SoftplusLoss",
                "loss_kwargs": {},
                "loss_kwargs_ranges": {},
                "regularizer": "NoRegularizer",
                "regularizer_kwargs": {},
                "regularizer_kwargs_ranges": {},
                "negative_sampler": "BasicNegativeSampler",
                "negative_sampler_kwargs": {},
                "negative_sampler_kwargs_ranges": {
                    "num_negs_per_pos": {"type": "int", "low": 1, "high": 50, "q": 1}
                },
                "evaluator": "RankBasedEvaluator",
                "evaluator_kwargs": {
                    "filtered": True,
                    "automatic_memory_optimization": True,
                },
                "evaluation_kwargs": {"batch_size": None},
                "training_kwargs": {"num_epochs": 1000, "label_smoothing": 0.0},
                "training_kwargs_ranges": {
                    "batch_size": {
                        "type": "int",
                        "low": 8,
                        "high": 11,
                        "scale": "power_two",
                    }
                },
                "stopper": "early",
                "stopper_kwargs": {
                    "frequency": 25,
                    "patience": 2,
                    "relative_delta": 0.002,
                },
                "n_trials": 100,
                "timeout": 129600,
                "metric": "hits@10",
                "direction": "maximize",
                "sampler": "random",
                "pruner": "nop",
            },
        }

    elif model_name == "CompGCN":
        config = {
            "optuna": {},
            "pipeline": {
                "model": "CompGCN",
                "model_kwargs_ranges": {
                    "embedding_dim": {
                        "type": "int",
                        "low": 6,
                        "high": 9,
                        "scale": "power_two",
                    }
                },
                "training_loop": "slcwa",
                "training_loop_kwargs": {"automatic_memory_optimization": True},
                "optimizer": "adam",
                "optimizer_kwargs": {"weight_decay": 0.0},
                "optimizer_kwargs_ranges": {
                    "lr": {"type": "float", "low": 0.0001, "high": 1.0, "scale": "log"}
                },
                "loss": "CrossEntropyLoss",
                "loss_kwargs": {},
                "loss_kwargs_ranges": {},
                "negative_sampler": "BasicNegativeSampler",
                "negative_sampler_kwargs": {},
                "negative_sampler_kwargs_ranges": {
                    "num_negs_per_pos": {"type": "int", "low": 1, "high": 50, "q": 1}
                },
                "evaluator": "RankBasedEvaluator",
                "evaluator_kwargs": {
                    "filtered": True,
                    "automatic_memory_optimization": True,
                },
                "evaluation_kwargs": {"batch_size": None},
                "training_kwargs": {"num_epochs": 1000, "label_smoothing": 0.0},
                "training_kwargs_ranges": {
                    "batch_size": {
                        "type": "int",
                        "low": 7,
                        "high": 11,
                        "scale": "power_two",
                    }
                },
                "stopper": "early",
                "stopper_kwargs": {
                    "frequency": 25,
                    "patience": 2,
                    "relative_delta": 0.002,
                },
                "n_trials": 100,
                "timeout": 129600,
                "metric": "hits@10",
                "direction": "maximize",
                "sampler": "random",
                "pruner": "nop",
            },
        }
    else:
        raise ValueError(
            f"config file for {model_name} was not provided and cannot be generated."
        )

    return config
