import base64
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from web.main import fig_to_b64, risk_label


def test_fig_to_b64_returns_valid_png():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    result = fig_to_b64(fig)
    assert isinstance(result, str)
    decoded = base64.b64decode(result)
    assert decoded[:4] == b'\x89PNG'


def test_fig_to_b64_closes_figure():
    fig, ax = plt.subplots()
    ax.plot([0], [0])
    open_before = len(plt.get_fignums())
    fig_to_b64(fig)
    assert len(plt.get_fignums()) < open_before


def test_risk_label_low():
    label, color = risk_label(0.1)
    assert label == "LOW"
    assert color == "green"


def test_risk_label_medium():
    label, color = risk_label(0.45)
    assert label == "MEDIUM"
    assert color == "amber"


def test_risk_label_high():
    label, color = risk_label(0.75)
    assert label == "HIGH"
    assert color == "red"


def test_risk_label_boundary_low_medium():
    label, color = risk_label(0.3)
    assert label == "MEDIUM"


def test_risk_label_boundary_medium_high():
    label, color = risk_label(0.6)
    assert label == "HIGH"
