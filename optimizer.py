# optimizer.py — Search-Based NS Parameter Optimization (Genetic Algorithm)
import random
import numpy as np
from losses import fitness_fn
from cast_negative_shadow import generate_shadow
from bev_to_driver import transform_bev_to_driver_view
from twinlitenet import run_twinlitenet
from hybridnet import run_hybridnets
from clrernet import run_clrernet
from cast_negative_shadow import CSV_PATH
from tqdm import tqdm 
import pandas as pd
# Hyperparameters
POP_SIZE = 20
MUTATION_RATE = 0.3
IMAGE_PATH = "./BEV.png"

# Parameter bounds (in pixels)
Y1_RANGE = (400, 475)
HEIGHT_RANGE = (5, 40)
RAND_X_RANGE = (50, 1266)
MIN_PIXEL_WIDTH = 2  # Approximately 4cm (≈ 2 pixels at given scale)

# ─────────────────────────────────────────────────────
def sample_candidate():
    while True:
        y1 = random.randint(*Y1_RANGE)
        height = random.randint(*HEIGHT_RANGE)
        if height < MIN_PIXEL_WIDTH:
            continue
        y2 = y1 + height
        upper_limit = 495 - height
        if y2 >= upper_limit:
            continue
        try:
            y3 = random.randint(y2, upper_limit)
        except ValueError:
            continue
        rand_x = random.randint(*RAND_X_RANGE)
        return (y1, y2, y3, rand_x)

# ─────────────────────────────────────────────────────
def mutate(candidate):
    y1, y2, y3, rand_x = candidate
    height = y2 - y1

    if random.random() < MUTATION_RATE:
        y1 = random.randint(*Y1_RANGE)
    if random.random() < MUTATION_RATE:
        height = max(MIN_PIXEL_WIDTH, random.randint(*HEIGHT_RANGE))
        y2 = y1 + height
    else:
        y2 = y1 + height

    if random.random() < MUTATION_RATE:
        upper_limit = 495 - height
        if y2 < upper_limit:
            try:
                y3 = random.randint(y2, upper_limit)
            except ValueError:
                return mutate(candidate)  # retry mutation
        else:
            return mutate(candidate)  # retry mutation

    if random.random() < MUTATION_RATE:
        rand_x = random.randint(*RAND_X_RANGE)

    return (y1, y2, y3, rand_x)

# ─────────────────────────────────────────────────────
def crossover(p1, p2):
    return tuple(random.choice([a, b]) for a, b in zip(p1, p2))

# ─────────────────────────────────────────────────────
# def evaluate_candidate(candidate, shadow_id, model):
#     try:
#         px = generate_shadow(IMAGE_PATH, shadow_id, *candidate)
#         px_drv = transform_bev_to_driver_view(px, shadow_id)
#         r1, o1 = run_twinlitenet(px_drv, shadow_id)
#         r2, o2 = run_hybridnets(px_drv, shadow_id)
#         r3, o3 = run_clrernet(model, px_drv, shadow_id)

#         from cast_negative_shadow import CSV_PATH
#         import pandas as pd
#         df = pd.read_csv(CSV_PATH)
#         row = df.loc[df['Image_Name'] == f"sun_cast_{shadow_id}.png"]
#         meta = row.iloc[0]
#         return fitness_fn(o1, o2, o3,
#                           meta['Length(m)'], meta['Width(cm)'], meta['Distance(cm)'], meta['Angle(deg)'])
#     except Exception as e:
#         print(f"⚠️ Error evaluating candidate {shadow_id}: {e}")
#         return float('-inf')


def evaluate_candidate(candidate, shadow_id, model):
    try:
        # 1. Inject and transform shadow
        px = generate_shadow(IMAGE_PATH, shadow_id, *candidate)
        if not px:  # injection failed
            print(f"⚠️ Empty shadow for ID {shadow_id}, skipping...")
            return float('-inf')

        
        px_drv = transform_bev_to_driver_view(px, shadow_id)

        # 2. Run all 3 models
        r1, o1 = run_twinlitenet(px_drv, shadow_id)
        r2, o2 = run_hybridnets(px_drv, shadow_id)
        r3, o3 = run_clrernet(model, px_drv, shadow_id)

        # 3. Load metadata from CSV

        df = pd.read_csv(CSV_PATH)
        row_index = df.index[df['Image_Name'] == f"sun_cast_{shadow_id}.png"]

        if not row_index.empty:
            idx = row_index[0]
            meta = df.loc[idx]

            # 4. Append detection columns
            df.loc[idx, 'twinlitenets'] = r1
            df.loc[idx, 'hybridnets'] = r2
            df.loc[idx, 'clrernet'] = r3
            df.to_csv(CSV_PATH, index=False)

            # 5. Compute fitness
            from losses import fitness_fn
            return fitness_fn(o1, o2, o3,
                              meta['Length(m)'], meta['Width(cm)'],
                              meta['Distance(cm)'], meta['Angle(deg)'])

        else:
            raise ValueError(f"No matching row found in CSV for shadow_id={shadow_id}")

    except Exception as e:
        print(f"⚠️ Error evaluating candidate {shadow_id}: {e}")
        return float('-inf')


# ─────────────────────────────────────────────────────
def run_search_optimization(n_samples, model):
    population = [sample_candidate() for _ in range(POP_SIZE)]
    generations = n_samples // POP_SIZE

    for gen in tqdm(range(generations), desc="🔁 Running Genetic Algorithm"):
        # print(f"\n=== Generation {gen+1} ===")
        scored = []
        for i, cand in enumerate(population):
            score = evaluate_candidate(cand, shadow_id=gen * POP_SIZE + i, model=model)
            # print(f"Candidate {i}: score = {score:.4f}")
            scored.append((score, cand))

        scored.sort(reverse=True)
        top_k = [cand for _, cand in scored[:POP_SIZE//2]]

        next_gen = top_k[:]
        while len(next_gen) < POP_SIZE:
            p1, p2 = random.sample(top_k, 2)
            child = mutate(crossover(p1, p2))
            next_gen.append(child)

        population = next_gen
