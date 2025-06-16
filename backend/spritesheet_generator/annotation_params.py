import asyncio
import json
import random
from typing import Any, Union


class AnnotationParams:
    MUST_ANNOTATE_ATTRIBUTES = [
        ("head", ["head"]),
        (
            "torso",
            [
                "clothes",
                "jacket",
                "vest",
                "armour",
            ],
        ),
        ("legs", ["legs"]),
        ("shoes", ["shoes"]),
        ("hair", ["hair"]),
    ]

    def __init__(self, chooser_to_available_options: dict[str, Any], seed: int = 0):
        # final response format: `attr_name=attr_string` (in chooser_to_available_options)
        random.seed(0)
        self.chooser_to_available_options = {
            k: chooser_to_available_options[k]
            for (attr_name, attrs) in self.MUST_ANNOTATE_ATTRIBUTES
            for k in attrs
        }
        self.lock = asyncio.Lock()

    async def get_to_annotate_attributes(self) -> list[str]:
        async with self.lock:
            return [
                random.choice(attrs)
                for (attr_name, attrs) in self.MUST_ANNOTATE_ATTRIBUTES
            ]

    async def get_to_annotate_options(
        self, attr_name: str, previous_annotations: list[str]
    ) -> Union[list[str], str]:
        async with self.lock:
            # head, clothes, jacket, vest, armour, legs, shoes, hair
            if attr_name not in self.chooser_to_available_options:
                raise ValueError(
                    f"attr_name {attr_name} not in chooser_to_available_options"
                )
            available_options = self.chooser_to_available_options[attr_name]
            for previous_annotation in previous_annotations:
                if isinstance(available_options, dict):
                    if previous_annotation not in available_options:
                        raise ValueError(
                            f"previous_annotation {previous_annotation} not in available_options {available_options}"
                        )
                    available_options = available_options[previous_annotation]
                else:
                    # has annotated the discrete attribute
                    # ATTENTION: remember that isinstance(str, list) is True
                    assert isinstance(
                        available_options, list
                    ), f"available_options must be a list, but got {type(available_options)}"
                    assert (
                        previous_annotation in available_options
                    ), f"previous_annotation {previous_annotation} not in available_options {available_options}"
                    final_annotation = previous_annotation
                    return f"{attr_name}={final_annotation}"
            return available_options
