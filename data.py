import numpy as np


def read_track_state_data(filename):
    """
    Read CVT track-state training data.

    Each line contains 89 values:

    Layer 1:
        xo, yo, zo, xe, ye, ze, mask

    Layer 2:
        xo, yo, zo, xe, ye, ze, mask

    ...

    Layer 12:
        xo, yo, zo, xe, ye, ze, mask

    Track state:
        d0, phi0, kappa, z0, tandip

    Total:
        12 * 7 + 5 = 89 values

    Returns
    -------
    features : np.ndarray
        shape [N, 84]

        For each layer:
            [xo, yo, zo, xe, ye, ze, mask]

    targets : np.ndarray
        shape [N, 5]

        Five track-state parameters.
    """

    features = []
    targets = []

    with open(filename, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() != ""
        ]

    for i, line in enumerate(lines):

        vals = [float(x) for x in line.split(",")]

        if len(vals) != 44:
            raise ValueError(
                f"Line {i+1}: expected 44 values "
                f"(39 input + 5 output), "
                f"got {len(vals)}"
            )

        # --------------------------------------------------
        # First 66 values:
        # 6 layers × 6 features
        # 6 layers × 5 features
        # --------------------------------------------------
        input_values = vals[:39]

        # --------------------------------------------------
        # Last 5 values:
        # track state
        # --------------------------------------------------
        state_values = vals[39:44]

        features.append(input_values)
        targets.append(state_values)

    features = np.array(
        features,
        dtype=np.float32
    )

    targets = np.array(
        targets,
        dtype=np.float32
    )

    return features, targets


# Example usage
if __name__ == "__main__":

    features, targets = read_track_state_data(
        "track_state.csv"
    )

    print(f"Read {len(features)} samples")

    print("\nInput shape :", features.shape)
    print("Target shape:", targets.shape)

    print("\nFirst sample:")

    print("input  =", features[0])
    print("target =", targets[0])