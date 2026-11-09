from copy import deepcopy

from gradlab.manual_evaluation import ManualEvaluationSupervisor


def test_manual_nonvisual_contract_remains_unmodified():
    contract = {
        "environment": {"env_provider": "gradlab", "game": "Bandit-v0"},
        "episodes": 100,
        "seed": 10000,
        "acceptance": [],
    }
    original = deepcopy(contract)
    assert ManualEvaluationSupervisor._current_protocol_contract(contract) == original
    assert contract == original
