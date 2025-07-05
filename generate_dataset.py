import numpy as np
import json
import os

def generate_ellipse(n_points=100, a=3, b=1, noise_std=0.1):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    points = np.vstack((x, y)).T
    points += np.random.normal(0, noise_std, points.shape)
    return points


def generate_rectangle(n_points=100, width=4, height=2, noise_std=0.1):
    points = []
    # Top edge
    points.extend(list(zip(np.linspace(-width / 2, width / 2, n_points // 4), np.full(n_points // 4, height / 2))))
    # Right edge
    points.extend(list(zip(np.full(n_points // 4, width / 2), np.linspace(height / 2, -height / 2, n_points // 4))))
    # Bottom edge
    points.extend(list(zip(np.linspace(width / 2, -width / 2, n_points // 4), np.full(n_points // 4, -height / 2))))
    # Left edge
    points.extend(list(zip(np.full(n_points // 4, -width / 2), np.linspace(-height / 2, height / 2, n_points // 4))))
    points = np.array(points)
    points += np.random.normal(0, noise_std, points.shape)
    return points


def generate_segment(n_points=100, length=4, noise_std=0.1):
    x = np.linspace(-length / 2, length / 2, n_points)
    y = np.zeros(n_points)
    points = np.vstack((x, y)).T
    points += np.random.normal(0, noise_std, points.shape)
    return points


def generate_cross(n_points=100, arm_length=2, noise_std=0.1):
    # Vertical arm
    p1 = np.vstack((np.zeros(n_points // 2), np.linspace(-arm_length, arm_length, n_points // 2))).T
    # Horizontal arm
    p2 = np.vstack((np.linspace(-arm_length, arm_length, n_points // 2), np.zeros(n_points // 2))).T
    points = np.vstack((p1, p2))
    points += np.random.normal(0, noise_std, points.shape)
    return points


def generate_shape_data(n_shapes_per_class=1000, n_points_per_shape=200, noise_bound=1.0, random_state=42):
    """
    Generates a fixed dataset of geometric shapes.
    """
    np.random.seed(random_state)

    all_shapes = []
    all_labels = []

    shape_generators = [generate_ellipse, generate_rectangle, generate_segment, generate_cross]

    for class_idx, generator in enumerate(shape_generators):
        for _ in range(n_shapes_per_class):
            # Use a random noise level for each shape up to the bound
            current_noise = np.random.uniform(0, noise_bound)
            shape = generator(n_points=n_points_per_shape, noise_std=current_noise)

            # Apply random rotation and translation
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])
            shape = shape @ rotation_matrix.T
            shape += np.random.uniform(-5, 5, size=2)

            all_shapes.append(shape)
            all_labels.append(class_idx)

    return all_shapes, np.array(all_labels)


if __name__ == "__main__":
    print("Generating synthetic dataset...")

    # --- Parameters for the dataset ---
    N_SHAPES_PER_CLASS = 1500
    N_POINTS_PER_SHAPE = 300
    NOISE_BOUND = 1.0
    RANDOM_STATE = 42

    # --- Generate data ---
    shapes, labels = generate_shape_data(
        n_shapes_per_class=N_SHAPES_PER_CLASS,
        n_points_per_shape=N_POINTS_PER_SHAPE,
        noise_bound=NOISE_BOUND,
        random_state=RANDOM_STATE
    )

    # --- Prepare for JSON serialization ---
    # Convert numpy arrays to lists
    shapes_as_lists = [shape.tolist() for shape in shapes]
    labels_as_list = labels.tolist()

    dataset = {
        'shapes': shapes_as_lists,
        'labels': labels_as_list,
        'metadata': {
            'n_shapes_per_class': N_SHAPES_PER_CLASS,
            'n_points_per_shape': N_POINTS_PER_SHAPE,
            'noise_bound': NOISE_BOUND,
            'random_state': RANDOM_STATE,
            'class_map': {0: 'Ellipse', 1: 'Rectangle', 2: 'Segment', 3: 'Cross'}
        }
    }

    # --- Save to file ---
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'dataset.json')
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset successfully generated and saved to '{output_path}'")
    print(f"Total shapes: {len(shapes)}")