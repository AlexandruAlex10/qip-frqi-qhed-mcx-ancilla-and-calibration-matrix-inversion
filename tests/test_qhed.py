
"""
Tests for QHED edge detection functions.

These tests cover:
- Output shape of edge maps
- Non-constant output for simple edge patterns
- Pipeline output structure
- Classical Sobel edge map output shape
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from src.qhed import baseline_qhed_edge_map, classical_sobel_edge_map, qhed_pipeline


def test_qhed_output_shape_4x4():
    img = np.array([
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 0],
        [255, 255, 0, 0],
    ], dtype=np.uint8)

    edge = baseline_qhed_edge_map(img)
    assert edge.shape == img.shape


def test_qhed_output_not_constant():
    img = np.zeros((8, 8), dtype=np.uint8)
    img[:, 4:] = 255
    edge = baseline_qhed_edge_map(img)
    assert np.std(edge) > 0


def test_qhed_pipeline_returns_expected_keys():
    img = np.zeros((4, 4), dtype=np.uint8)
    out = qhed_pipeline(img)
    assert set(out.keys()) == {"statevector", "reconstructed", "edge_map"}


def test_classical_sobel_output_shape():
    img = np.zeros((8, 8), dtype=np.uint8)
    img[2:6, 2:6] = 255
    edge = classical_sobel_edge_map(img)
    assert edge.shape == img.shape
