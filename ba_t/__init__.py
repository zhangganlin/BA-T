"""BA-T: an iterative transformer for two-view bundle adjustment (inference code).

Subpackages / modules:
    models     -- BAT model (ViT encoder, recursive BA layer, point / pose heads, RoPE)
    data       -- load images from paths / folders with the training preprocessing
    inference  -- checkpoint loading, pair inference and multi-view inference
    graph      -- view graph (covisible views) from encoder-feature similarity
    pose_graph -- global camera poses from the relative poses of a view graph
    utils      -- cropping / image normalization helpers
"""
