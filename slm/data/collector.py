"""
slm.data.collector — Ingestion and streaming collectors for pre-training corpora.

Supports streaming ingestion from Hugging Face Datasets without downloading
entire gigabyte-scale datasets locally, as well as local file ingestion and
synthetic corpus generation for offline overfit tests.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from slm.core.exceptions import DataCollectionError
from slm.core.logging import get_logger
from slm.data.schema import Document, Domain, LicenseType
from slm.data.sources import SourceConfig

logger = get_logger("data.collector")


class BaseCollector(ABC):
    """Abstract base collector for data ingestion."""

    @abstractmethod
    def stream_documents(self) -> Generator[Document, None, None]:
        """Yield documents one by one as they are fetched or read."""
        pass

    def save_raw(self, output_path: Path | str, max_docs: Optional[int] = None) -> int:
        """
        Save streamed documents to a JSONL file.

        Returns
        -------
        int
            Number of documents successfully written.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for doc in self.stream_documents():
                f.write(doc.to_json_line() + "\n")
                count += 1
                if max_docs is not None and count >= max_docs:
                    break

        logger.info(f"Saved {count} raw documents to {out_file}")
        return count


class LocalFileCollector(BaseCollector):
    """
    Ingests text or JSONL files from a local directory or file path.
    """

    def __init__(
        self,
        path: Path | str,
        source_name: str = "local_text",
        domain: Domain = Domain.GENERAL_KNOWLEDGE,
        license_type: LicenseType = LicenseType.CC0,
        glob_pattern: str = "*.txt",
    ) -> None:
        self.path = Path(path)
        self.source_name = source_name
        self.domain = domain
        self.license = license_type
        self.glob_pattern = glob_pattern

    def stream_documents(self) -> Generator[Document, None, None]:
        if not self.path.exists():
            raise DataCollectionError(
                f"Path does not exist: {self.path}",
                suggestion="Verify the data path exists or run download scripts first."
            )

        if self.path.is_file():
            files = [self.path]
        else:
            files = list(self.path.glob(self.glob_pattern))

        for file_path in files:
            if file_path.suffix == ".jsonl":
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield Document.from_json_line(line)
                        except Exception as e:
                            logger.warning(f"Skipping malformed JSON line in {file_path}: {e}")
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read().strip()
                        if text:
                            yield Document(
                                text=text,
                                source=self.source_name,
                                domain=self.domain,
                                license=self.license,
                                metadata={"filename": file_path.name}
                            )
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")


class SyntheticOverfitCollector(BaseCollector):
    """
    Generates a high-quality deterministic synthetic corpus across stories,
    arithmetic, Python code, and Q&A.

    Essential for:
    1. Phase 14 tiny model overfit validation (memorization test).
    2. Fast offline development and testing of tokenizers, data filters, and data loaders
       without requiring internet downloads or external API keys.
    """

    SAMPLE_STORIES = [
        "Once upon a time, there was a little blue bird named Pip. Pip lived in a high pine tree. Every morning, Pip sang a cheerful song for all the animals in the forest.",
        "Mia had a bright red bicycle. She rode her bicycle to the town market every Saturday. On the way, she stopped to greet Mr. Green, the friendly baker.",
        "A wise old owl sat on an oak branch. The moon shone brightly through the leaves. The little forest mice gathered to listen to the owl's gentle stories.",
        "Leo was a small puppy with floppy brown ears. He loved chasing autumn leaves as they danced in the cool wind across the park lawn.",
        "In a quiet valley between two green hills, there was a crystal-clear stream. Small silver fish swam against the gentle current all day long.",
    ]

    SAMPLE_CODE = [
        "def factorial(n: int) -> int:\n    \"\"\"Calculate the factorial of a non-negative integer.\"\"\"\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
        "def fibonacci(n: int) -> list[int]:\n    \"\"\"Return the first n Fibonacci numbers.\"\"\"\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq",
        "def is_prime(num: int) -> bool:\n    \"\"\"Check if an integer is prime.\"\"\"\n    if num <= 1:\n        return False\n    for i in range(2, int(num ** 0.5) + 1):\n        if num % i == 0:\n            return False\n    return True",
        "class Stack:\n    \"\"\"A simple Last-In-First-Out stack implementation.\"\"\"\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        if not self._items:\n            raise IndexError('pop from empty stack')\n        return self._items.pop()\n    def is_empty(self) -> bool:\n        return len(self._items) == 0",
        "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Find target index in sorted array or -1 if absent.\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
    ]

    SAMPLE_MATH = [
        "Problem: Maya has 15 red apples and 9 green apples. She gives 6 apples to her brother and 4 apples to her friend. How many apples does Maya have left?\nSolution: Total apples = 15 + 9 = 24. Total given away = 6 + 4 = 10. Remaining apples = 24 - 10 = 14.",
        "Problem: A train travels at a speed of 60 miles per hour. How far does the train travel in 3.5 hours?\nSolution: Distance = Speed * Time = 60 * 3.5 = 210 miles.",
        "Problem: A rectangle has a length of 12 cm and a width of 7 cm. What is its perimeter and area?\nSolution: Perimeter = 2 * (12 + 7) = 38 cm. Area = 12 * 7 = 84 square cm.",
        "Problem: Solve for x: 3x + 15 = 45.\nSolution: Subtract 15 from both sides: 3x = 30. Divide by 3: x = 10.",
        "Problem: If a box contains 8 red balls, 5 blue balls, and 7 green balls, what is the probability of drawing a blue ball at random?\nSolution: Total balls = 8 + 5 + 7 = 20. Probability = 5 / 20 = 1/4 or 25%.",
    ]

    SAMPLE_QA = [
        "Question: What is photosynthesis?\nAnswer: Photosynthesis is the process by which green plants and some organisms use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
        "Question: Explain the difference between speed and velocity.\nAnswer: Speed is a scalar quantity measuring how fast an object moves regardless of direction. Velocity is a vector quantity specifying both speed and direction.",
        "Question: What are the primary colors of light?\nAnswer: The primary colors of light in the additive color system are red, green, and blue (RGB).",
        "Question: Why do we have seasons on Earth?\nAnswer: Earth has seasons because its axis of rotation is tilted at about 23.5 degrees relative to its orbital plane around the Sun.",
        "Question: What is the purpose of an operating system?\nAnswer: An operating system manages computer hardware and software resources, providing common services and abstractions for computer programs.",
    ]

    def __init__(self, repeats: int = 20) -> None:
        self.repeats = repeats

    def stream_documents(self) -> Generator[Document, None, None]:
        for rep in range(self.repeats):
            for i, story in enumerate(self.SAMPLE_STORIES):
                yield Document(
                    text=story,
                    source="synthetic_stories",
                    domain=Domain.STORIES_LITERATURE,
                    license=LicenseType.CC0,
                    metadata={"index": i, "repeat": rep}
                )
            for i, code in enumerate(self.SAMPLE_CODE):
                yield Document(
                    text=code,
                    source="synthetic_python",
                    domain=Domain.CODE_PYTHON,
                    license=LicenseType.MIT,
                    metadata={"index": i, "repeat": rep}
                )
            for i, math in enumerate(self.SAMPLE_MATH):
                yield Document(
                    text=math,
                    source="synthetic_math",
                    domain=Domain.MATH_REASONING,
                    license=LicenseType.CC0,
                    metadata={"index": i, "repeat": rep}
                )
            for i, qa in enumerate(self.SAMPLE_QA):
                yield Document(
                    text=qa,
                    source="synthetic_qa",
                    domain=Domain.CONVERSATION_QA,
                    license=LicenseType.CC0,
                    metadata={"index": i, "repeat": rep}
                )


class StreamingHFCollector(BaseCollector):
    """
    Streams records from Hugging Face Datasets in streaming mode.

    Does NOT download multi-gigabyte files to disk beforehand.
    """

    def __init__(self, source_config: SourceConfig) -> None:
        self.config = source_config

    def stream_documents(self) -> Generator[Document, None, None]:
        if not self.config.hf_dataset_id:
            raise DataCollectionError(
                f"Source '{self.config.name}' does not specify a Hugging Face dataset ID."
            )

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise DataCollectionError(
                "datasets library is required to stream from Hugging Face.",
                suggestion="Run: pip install datasets"
            ) from e

        logger.info(
            f"Streaming from Hugging Face: {self.config.hf_dataset_id} "
            f"(subset: {self.config.hf_subset}, split: {self.config.hf_split})"
        )

        try:
            kwargs = {
                "split": self.config.hf_split,
                "streaming": True,
            }
            if self.config.hf_subset:
                ds = load_dataset(self.config.hf_dataset_id, self.config.hf_subset, **kwargs)
            else:
                ds = load_dataset(self.config.hf_dataset_id, **kwargs)

            for record in ds:
                text = record.get(self.config.text_column)
                if text and isinstance(text, str) and text.strip():
                    yield Document(
                        text=text.strip(),
                        source=self.config.name,
                        domain=self.config.domain,
                        license=self.config.license,
                        metadata={
                            "hf_dataset": self.config.hf_dataset_id,
                            "hf_split": self.config.hf_split,
                        }
                    )
        except Exception as exc:
            raise DataCollectionError(
                f"Failed to stream from Hugging Face dataset {self.config.hf_dataset_id}: {exc}",
                suggestion="Check internet connectivity or authentication if using gated datasets."
            ) from exc
