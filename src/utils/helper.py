def calculate_scale_pos_weight(y_train):
    """
    Calculate the scale_pos_weight for imbalanced datasets.
    
    Args:
        y_train (array-like): The target variable of the training set.
    Returns:
        float: The scale_pos_weight value.
    """
    return (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0