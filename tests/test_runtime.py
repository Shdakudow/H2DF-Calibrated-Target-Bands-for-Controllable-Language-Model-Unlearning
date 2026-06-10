import json

from h2df.runtime import make_dataset


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [ord(char) % 20 for char in text]
        if add_special_tokens:
            ids = [1] + ids
        return ids[:max_length] if truncation and max_length else ids


def test_make_dataset_can_use_distinct_evaluation_source(tmp_path):
    retain_path = tmp_path / "retain1.jsonl"
    retain_eval_path = tmp_path / "retain2.jsonl"
    retain_path.write_text(json.dumps({"text": "training retain"}) + "\n")
    retain_eval_path.write_text(json.dumps({"text": "evaluation retain"}) + "\n")
    config = {
        "experiment": {"task": "muse"},
        "data": {
            "retain": {"path": str(retain_path)},
            "retain_eval": {"path": str(retain_eval_path)},
            "text_column": "text",
        },
        "training": {"max_length": 64},
    }

    records, _ = make_dataset(
        config,
        FakeTokenizer(),
        "retain",
        source_name="retain_eval",
    )

    assert records[0]["text"] == "evaluation retain"
