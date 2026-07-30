from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def generate_model_summary_image(
    output_path: Optional[Path | str] = None,
    model_name: str = "productivity2/resnet18.onnx",
    architecture: str = "ResNet-18",
    layers: Optional[Sequence[Tuple[str, str, str]]] = None,
    param_count: str = "11.7M",
    classes: Optional[Sequence[str]] = None,
) -> bool:
    """Render a simple JPG summary image for the model architecture and parameter counts."""
    if output_path is None:
        output_path = Path(__file__).with_name("model_summary.jpg")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if layers is None:
        layers = [
            ("Input", "224x224x3", "0"),
            ("Conv1", "112x112x64", "9.4K"),
            ("Layer1", "56x56x64", "0.15M"),
            ("Layer2", "28x28x128", "0.53M"),
            ("Layer3", "14x14x256", "2.1M"),
            ("Layer4", "7x7x512", "7.1M"),
            ("FC", "1000", "0.5M"),
        ]

    width, height = 1800, 1000
    image = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

    cv2.putText(
        image,
        "model summary",
        (60, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"model: {model_name}",
        (60, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"architecture: {architecture}",
        (60, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"parameters: {param_count}",
        (60, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    start_x = 80
    start_y = 240
    line_gap = 85

    for idx, (name, shape, params) in enumerate(layers):
        y = start_y + idx * line_gap
        cv2.putText(image, f"{name}: {shape} | {params}", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(image, (70, y + 10), (720, y + 10), (0, 0, 0), 1)

    success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return success


if __name__ == "__main__":
    output_path = Path(__file__).with_name("model_summary.jpg")
    generated = generate_model_summary_image(output_path=output_path)
    print(f"Generated {output_path} -> {generated}")
