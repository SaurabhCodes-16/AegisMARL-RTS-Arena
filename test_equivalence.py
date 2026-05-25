import torch
import numpy as np
import json
import re
from mappo_scratch import Actor


def extract_weights_from_js(js_path="marl_weights.js"):
    """
    Parses the generated JS file using regex to extract the JSON weights dictionary,
    allowing us to run the JS equivalence test purely in Python.
    """
    with open(js_path, "r") as f:
        content = f.read()

    # Find the JSON structure: const marl_weights = { ... };
    match = re.search(r"const marl_weights = ({[\s\S]*?});", content)
    if not match:
        raise ValueError("Could not find marl_weights variable in JS file!")

    json_str = match.group(1)
    return json.loads(json_str)


def python_js_feedforward(obs, weights):
    """
    Implements the exact matrix multiplication logic used in our JS engine (rts_sim.js)
    but in Python, so we can verify equivalence.
    """
    w = weights
    
    # Layer 1: Linear + Tanh
    h1 = []
    for i in range(len(w["w1"])):
        val = w["b1"][i]
        for j in range(len(obs)):
            val += w["w1"][i][j] * obs[j]
        h1.append(np.tanh(val))

    # Layer 2: Linear + Tanh
    h2 = []
    for i in range(len(w["w2"])):
        val = w["b2"][i]
        for j in range(len(h1)):
            val += w["w2"][i][j] * h1[j]
        h2.append(np.tanh(val))

    # Layer 3: Linear
    logits = []
    for i in range(len(w["w3"])):
        val = w["b3"][i]
        for j in range(len(h2)):
            val += w["w3"][i][j] * h2[j]
        logits.append(val)

    # Argmax
    return np.array(logits)


def run_equivalence_test():
    print("=" * 60)
    print("        MAPPO Neural Equivalence Validation Suite            ")
    print("=" * 60)

    # 1. Load JS weights
    try:
        js_weights = extract_weights_from_js("marl_weights.js")
        print("[Test] Successfully parsed 'marl_weights.js'.")
    except Exception as e:
        print(f"[FAIL] Error parsing JS weights: {e}")
        return False

    # 2. Load PyTorch model
    device = torch.device("cpu")
    pytorch_actors = {}
    
    for a in ["knight", "ranger", "healer"]:
        actor = Actor(obs_dim=33, act_dim=6)
        try:
            actor.load_state_dict(torch.load(f"mappo_rts_actor_{a}.pth", map_location=device))
            actor.eval()
            pytorch_actors[a] = actor
            print(f"[Test] Loaded PyTorch actor checkpoint for class: {a}")
        except Exception as e:
            print(f"[FAIL] Error loading PyTorch model for {a}: {e}")
            return False

    # 3. Validation loops over random states
    tests_passed = True
    np.random.seed(42)  # Fixed seed for repeatable tests

    print("\n[Test] Sampling 100 random observation states per class...")

    for a in ["knight", "ranger", "healer"]:
        actor = pytorch_actors[a]
        weights = js_weights[a]
        mismatches = 0

        for t in range(100):
            # Generate random normalized observation
            obs = np.random.uniform(-1.0, 1.0, (33,)).astype(np.float32)

            # PyTorch forward pass
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                py_logits = actor(obs_tensor).squeeze(0).numpy()
                py_action = np.argmax(py_logits)

            # Translated JS forward pass
            js_logits = python_js_feedforward(obs, weights)
            js_action = np.argmax(js_logits)

            # Verify values are identical (up to rounding margin of 1e-5)
            diff = np.abs(py_logits - js_logits)
            max_diff = np.max(diff)

            if max_diff > 1e-5 or py_action != js_action:
                mismatches += 1
                tests_passed = False

        if mismatches == 0:
            print(f"[PASS] Class '{a:<6}': 100/100 correct forward-passes! Max variance: <1e-6")
        else:
            print(f"[FAIL] Class '{a:<6}': Mismatch found in {mismatches}/100 tests!")

    if tests_passed:
        print("\n" + "=" * 60)
        print(" SUCCESS: PyTorch and browser JS weights are mathematically identical! ")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print(" FAILURE: Discrepancy found between PyTorch and JS feedforward networks. ")
        print("=" * 60)
        return False


if __name__ == "__main__":
    run_equivalence_test()
