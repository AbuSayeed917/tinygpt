import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOOD_PARAGRAPH = (
    "The river runs through the old town and past the mill. In spring the water "
    "rises with the melting snow, and the children of the village gather on the "
    "bridge to watch the boats. Farmers bring their goods to the market that "
    "stands beside the church, and the bakers open early to sell bread to the "
    "workers who have walked in from the hills with their tools."
)


@pytest.fixture
def good_paragraph() -> str:
    return GOOD_PARAGRAPH
