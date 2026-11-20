import torch

from app.llm import LoRALinear, generate, train_lora
from app.storage import append_record, load_records


def test_lora_layer_shape_and_trainable_parameters():
    layer = LoRALinear(5, 7, rank=2)
    output = layer(torch.randn(3, 5))
    assert output.shape == (3, 7)
    assert not layer.base.weight.requires_grad
    assert layer.lora_A.requires_grad and layer.lora_B.requires_grad


def test_tiny_training_and_generation():
    trained = train_lora(steps=2)
    result = generate(trained, "[Soccer] Goal! Commentary:", max_new_tokens=20, seed=1)
    assert isinstance(result, str) and result
    assert trained.training_loss > 0


def test_jsonl_storage_round_trip(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    append_record(path, {"event": "goal", "commentary": "What a finish!"})
    assert load_records(path) == [{"event": "goal", "commentary": "What a finish!"}]

