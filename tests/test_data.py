from h2df.data import CausalCollator, tokenize_record


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [ord(char) % 20 for char in text]
        if add_special_tokens:
            ids = [1] + ids
        return ids[:max_length] if truncation and max_length else ids


def test_tofu_masks_prompt_tokens():
    result = tokenize_record(
        {"question": "Q", "answer": "A", "question_type": "other", "domain": "default"},
        FakeTokenizer(),
        "tofu",
        max_length=32,
        prompt_template="{question}:",
    )
    first_label = next(index for index, value in enumerate(result["labels"]) if value != -100)
    assert first_label > 0
    assert result["labels"][first_label:] == result["input_ids"][first_label:]


def test_collator_pads_labels_with_ignore_index():
    collator = CausalCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2], "index": 0},
            {"input_ids": [1], "attention_mask": [1], "labels": [1], "index": 1},
        ]
    )
    assert batch["labels"].tolist() == [[1, 2], [1, -100]]
