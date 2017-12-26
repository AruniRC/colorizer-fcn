


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    ),

    1: dict(
        max_iteration=500000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=1000,
    ),

    2: dict(
        max_iteration=500000,
        lr=1.0e-12,    # lower learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=1000,
    ),

    3: dict(
        max_iteration=500000,
        lr=1.0e-10,    # lower learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=1000,
    ),

    # debugging on tiny dataset of 9 images
    4: dict(
        max_iteration=500000,
        lr=1.0e-10,    # lower learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=5,
    ),

    # debugging on tiny dataset of 9 images
    # DEBUG: use MSE loss
    5: dict(
        max_iteration=10000,
        lr=1.0e-2,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=5,
    ),

    6: dict(
        max_iteration=10000,
        lr=1.0e-10,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=10,
    ),

    7: dict(
        max_iteration=10000,
        lr=1.0e-8,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=50,
    ),

    8: dict(
        max_iteration=10000,
        lr=1.0e-10,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=50,
    ),

    9: dict(
        max_iteration=1000,
        lr=1.0e-5,     # changed learning rate
        momentum=0.9,  # momentum lowered
        weight_decay=0.0005,
        interval_validate=10,
    )
}