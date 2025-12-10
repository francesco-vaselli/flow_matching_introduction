# Flow Matching Exercise Setup

## Overview

This directory contains an interactive, hands-on exercise on Flow Matching and Diffusion Models. Students implement core algorithms while we provide the infrastructure (neural networks, training loops, visualizations).

## Files

### Main Notebook
- **`solution.ipynb`** - The exercise notebook with:
  - Comprehensive introduction and learning objectives
  - Theory explanations with mathematical formulas
  - 5 functions for students to implement (with TODO comments)
  - Provided code: NN architecture, training loops, visualization
  - Test cells to validate implementations
  - CMS jet simulation application

### Test Suite
- **`test_exercises.py`** - Comprehensive test suite with:
  - 5 test classes (one per function to implement)
  - Multiple test cases per function (boundary conditions, shapes, numerical properties)
  - Clear pass/fail feedback with helpful error messages
  - `run_tests()` function to validate implementations

### Supporting Files
- **`utils.py`** - Utility functions (data loading, plotting helpers)
- **`figs/`** - Figures used in the notebook
- **`data.npy`** - Sample data for exercises

## What Students Implement

Students implement **5 core functions** that are fundamental to flow matching and diffusion:

### Exercise 1: Conditional Paths (Part 1.3.2)

1. **`conditional_path(x_0, x_1, t)`**
   - Implements affine interpolation: $x_t = (1-t)x_0 + tx_1$
   - Tests: boundary conditions, midpoint, batch processing, linearity

2. **`conditional_vector_field(x_0, x_1, t)`**
   - Implements velocity: $v_t = x_1 - x_0$
   - Tests: affine velocity, time independence, batch processing

### Exercise 2: Sampling Functions (Part 2.4)

3. **`vector_field_to_score(model, t, x, eps)`**
   - Converts flow to score: $s_\theta(t,x) = \frac{t \cdot u_\theta(t,x) - x}{1-t}$
   - Tests: formula correctness, numerical stability, shape preservation

4. **`sample_with_flow(model, n_samples, n_steps)`**
   - Deterministic ODE sampling using Euler method
   - Tests: output shape, deterministic behavior, initial conditions

5. **`sample_with_diffusion(model, n_samples, n_steps, sigma)`**
   - Stochastic SDE sampling using Euler-Maruyama method
   - Tests: output shape, stochastic behavior, sigma effects

## What's Provided

To focus learning on core concepts, we provide:

- ✅ **Neural Network Architecture**
  - `MLP` class with Swish activations
  - Proper time embedding
  - 4-layer architecture with configurable hidden dimensions

- ✅ **Training Infrastructure**
  - `train_flow_matching()` function
  - Flow matching loss computation
  - Optimization loop with progress tracking

- ✅ **Visualization Code**
  - Distribution plotting
  - Trajectory visualization
  - Training curve plotting
  - Comparison plots for flow vs. diffusion

- ✅ **CMS Application**
  - Data loading and preprocessing
  - Feature engineering
  - Model training for jet simulation
  - Validation and evaluation code

## Notebook Structure

### Part 1: Building Intuition
- Theory: Vector fields and probability paths
- **Exercise 1.3.2**: Implement `conditional_path()` and `conditional_vector_field()`
- Test cell validates implementations
- Visualizations show paths and velocities

### Part 2: Flow Matching vs Diffusion
- Theory: Connection between flow matching and diffusion models
- Training on simple distributions (provided)
- **Exercise 2.4**: Implement score conversion and sampling functions
- Test cell validates all implementations
- Compare flow vs. diffusion sampling with visualizations

### Part 3: CMS Jet Simulation
- Real-world application (all code provided)
- Students observe their implementations in action
- Train model to simulate detector-level jet features

## How to Use

### For Students

1. **Read the introduction** (first cell) - explains exercise format and expectations
2. **Read the Quick Reference Card** - handy formulas and tips
3. **Work through Part 1** - implement conditional path functions
4. **Run test cells** after each implementation
5. **Fix issues** if tests fail (error messages guide you)
6. **Move to Part 2** once tests pass
7. **Complete sampling functions** and validate with tests
8. **Observe results** in the CMS application

### For Instructors

1. Students work through `solution.ipynb` implementing functions
2. Tests provide immediate feedback
3. After completion, you can provide `exercises.ipynb` (with solutions) for reference
4. Estimated time: 1.5-2 hours depending on background

## Testing System

The test suite (`test_exercises.py`) provides:

- **Comprehensive coverage**: Multiple test cases per function
- **Clear feedback**: ✅ Pass, ❌ Fail with error message, ⚠️ Runtime error
- **Incremental testing**: Can test functions individually or all together
- **Educational messages**: Tests explain what went wrong

### Running Tests

In the notebook:
```python
from test_exercises import run_tests

# Test just the first two functions
run_tests(
    conditional_path=conditional_path,
    conditional_vector_field=conditional_vector_field
)

# Test all functions
run_tests(
    conditional_path=conditional_path,
    conditional_vector_field=conditional_vector_field,
    vector_field_to_score=vector_field_to_score,
    sample_with_flow=sample_with_flow,
    sample_with_diffusion=sample_with_diffusion
)
```

## Learning Outcomes

After completing this exercise, students will:

1. **Understand** the mathematical foundations of flow matching
2. **Implement** probability paths and vector fields from scratch
3. **Connect** flow matching to diffusion models mathematically
4. **Code** both deterministic (ODE) and stochastic (SDE) sampling
5. **Apply** these concepts to real physics simulation problems
6. **Gain intuition** through visualization of paths and trajectories

## Tips for Success

- 📖 **Read the math carefully** - each formula is explained with intuition
- 🎯 **Start simple** - implement the basic formula, then handle edge cases
- 💡 **Use the hints** - TODO comments guide you through implementation
- 🧪 **Test frequently** - catch bugs early before moving forward
- 📊 **Check visualizations** - plots help verify correctness

## Common Issues

See the "Common Issues & Solutions" section in the Quick Reference Card cell.

## Requirements

```
torch >= 1.9.0
numpy >= 1.19.0
matplotlib >= 3.3.0
```

For Part 3 (CMS application), also need:
```
flow_matching (pip install flow-matching)
sklearn
```

## References

- [Flow Matching Blog Post](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- [Flow Matching Paper](https://arxiv.org/abs/2210.02747)
- [Diffusion Models Tutorial](https://arxiv.org/abs/2208.11970)

## Contact

For questions or issues, please contact the course instructors.
