# losses.py — Fitness Function for NS Search

def fitness_fn(overlap1, overlap2, overlap3,
               length_m, width_cm, distance_cm, angle_deg):
    """
    Composite fitness function for evaluating the quality of a shadow polygon.
    Higher score = better attack (maximize this).

    Parameters
    ----------
    overlap1 : float
        Overlap ratio from TwinLiteNet (0 to 1)
    overlap2 : float
        Overlap ratio from HybridNet (0 to 1)
    overlap3 : float
        Overlap ratio from CLRerNet (0 to 1)
    length_m : float
        Length of the shadow in meters
    width_cm : float
        Width of the shadow in centimeters
    distance_cm : float
        Distance from real lane in centimeters
    angle_deg : float
        Angle of the shadow in degrees

    Returns
    -------
    float
        Fitness score (higher is better for maximization-based search)
    """
    # Detection quality: average overlap ratio
    detection_score = (overlap1 + overlap2 + overlap3) / 3

    # Normalize geometric factors (heuristics for realism)
    norm_len = (length_m - 1) / (40 - 1)                  # prefer long
    norm_wid = 1 - abs((width_cm - 15) / 10)              # prefer ~15cm lane width
    norm_dst = 1 - min(distance_cm / 350, 1.0)            # prefer closer to lane
    norm_ang = 1 - abs(angle_deg / 90)                   # prefer small angle

    # Geometric realism score (average of normalized heuristics)
    geom_score = (norm_len + norm_wid + norm_dst + norm_ang) / 4

    # Weighted sum: more weight to fooling detectors
    return 0.7 * detection_score + 0.3 * geom_score
