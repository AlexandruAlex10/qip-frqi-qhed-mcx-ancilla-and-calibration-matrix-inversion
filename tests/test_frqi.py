
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.frqi import (
    build_frqi_statevector,
    reconstruct_image_from_statevector,
    required_position_qubits,
    validate_grayscale_image,
    image_to_angles,
    l2_error,
)


def test_validate_grayscale_image_rejects_non_square():
    with pytest.raises(ValueError):
        validate_grayscale_image(np.zeros((4, 8), dtype=np.uint8))


def test_required_position_qubits():
    assert required_position_qubits(4) == 4
    assert required_position_qubits(8) == 6


def test_image_to_angles_bounds():
    img = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    theta = image_to_angles(img)
    assert theta.min() >= 0
    assert theta.max() <= np.pi / 2 + 1e-12


def test_frqi_round_trip_4x4():
    img = np.array([
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 0],
        [255, 255, 0, 0],
    ], dtype=np.uint8)

    state = build_frqi_statevector(img)
    recon = reconstruct_image_from_statevector(state, 4)
    assert recon.shape == img.shape
    assert l2_error(img, recon) <= 1.0


def test_frqi_round_trip_8x8():
    img = np.zeros((8, 8), dtype=np.uint8)
    img[:, 4:] = 200
    img[2:6, 2:6] = 255
    state = build_frqi_statevector(img)
    recon = reconstruct_image_from_statevector(state, 8)
    assert recon.shape == img.shape
    assert l2_error(img, recon) <= 1.0
