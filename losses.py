# losses.py — Fitness Function for NS Search (Unbiased Version)

def fitness_fn(r1, r2, r3,
               overlap1, overlap2, overlap3,
               length_m, width_cm, distance_cm, angle_deg):
    """
    Fitness function for evaluating the effectiveness of a shadow polygon.
    Combines binary detection outcomes and overlap quality from three lane detection models.

    Parameters
    ----------
    r1 : bool
        Detection result from TwinLiteNet (True = NS detected as lane)
    r2 : bool
        Detection result from HybridNets (True = NS detected as lane)
    r3 : bool
        Detection result from CLRerNet (True = NS detected as lane)
    overlap1 : float
        Overlap ratio with TwinLiteNet (0 to 1)
    overlap2 : float
        Overlap ratio with HybridNets (0 to 1)
    overlap3 : float
        Overlap ratio with CLRerNet (0 to 1)
    length_m : float
        Length of the shadow in meters (logged for analysis only)
    width_cm : float
        Width of the shadow in centimeters (logged for analysis only)
    distance_cm : float
        Distance from the real lane in centimeters (logged for analysis only)
    angle_deg : float
        Angle of the shadow in degrees (logged for analysis only)

    Returns
    -------
    float
        Fitness score (higher = better attack)
    """

    # Objective 1: Proportion of models fooled (True = 1, False = 0)
    detection_score = (int(r1) + int(r2) + int(r3)) / 3

    # Objective 2: Average overlap across all 3 models
    overlap_score = (overlap1 + overlap2 + overlap3) / 3

    # Combined score — adjust weights as needed
    return 0.5 * detection_score + 0.5 * overlap_score
