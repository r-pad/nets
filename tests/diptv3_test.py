import pytest
import torch

from rpad.nets.diptv3 import DiPTv3, DiPTv3Adapter


# Skip if cuda not detected.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_diptv3():
    # Create a model
    final_dimension = 11
    B = 3
    N = 256
    model = DiPTv3Adapter(
        model=DiPTv3(in_channels=6), final_dimension=final_dimension
    ).cuda()

    # Create a random input tensor
    x = torch.rand(B, 3, N).cuda().float()
    t = (
        torch.rand(
            B,
        )
        .cuda()
        .float()
    )
    x0 = torch.rand(B, 3, N).cuda().float()

    # Run the model
    output = model(x, t, x0)

    breakpoint()

    # Check the output shape
    assert output.shape == (B, final_dimension, N)
