"""
Test Suite for Flow Matching Exercises

This test suite validates the implementations of key flow matching and diffusion functions.
Students should implement the functions and run these tests to verify correctness.

Tests cover:
1. Conditional probability path (affine interpolation)
2. Conditional vector field (velocity computation)
3. Vector field to score function conversion
4. Flow-based sampling (ODE solver)
5. Diffusion-based sampling (SDE solver)
"""

import torch
import numpy as np


class TestConditionalPath:
    """Test the conditional_path function"""
    
    def test_boundary_conditions(self, conditional_path):
        """Test that path satisfies boundary conditions: x(0)=x_0, x(1)=x_1"""
        x_0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        
        # At t=0, should return x_0
        x_t0 = conditional_path(x_0, x_1, 0.0)
        assert torch.allclose(x_t0, x_0, atol=1e-6), "At t=0, path should return x_0"
        
        # At t=1, should return x_1
        x_t1 = conditional_path(x_0, x_1, 1.0)
        assert torch.allclose(x_t1, x_1, atol=1e-6), "At t=1, path should return x_1"
    
    def test_midpoint(self, conditional_path):
        """Test that midpoint is correct (affine interpolation)"""
        x_0 = torch.tensor([[0.0, 0.0]])
        x_1 = torch.tensor([[10.0, 20.0]])
        
        # At t=0.5, should be midpoint
        x_mid = conditional_path(x_0, x_1, 0.5)
        expected = torch.tensor([[5.0, 10.0]])
        assert torch.allclose(x_mid, expected, atol=1e-6), "At t=0.5, should return midpoint"
    
    def test_batch_processing(self, conditional_path):
        """Test that function handles batches correctly"""
        batch_size = 100
        x_0 = torch.randn(batch_size, 2)
        x_1 = torch.randn(batch_size, 2)
        t = torch.rand(batch_size, 1)
        
        x_t = conditional_path(x_0, x_1, t)
        
        # Check shape
        assert x_t.shape == (batch_size, 2), "Output shape should match input"
        
        # Verify interpolation for a few samples
        for i in range(min(5, batch_size)):
            expected = (1 - t[i]) * x_0[i] + t[i] * x_1[i]
            assert torch.allclose(x_t[i], expected, atol=1e-5), f"Batch item {i} incorrect"
    
    def test_linearity(self, conditional_path):
        """Test that path is linear (affine)"""
        x_0 = torch.tensor([[1.0, 2.0]])
        x_1 = torch.tensor([[5.0, 10.0]])
        
        # Check several points along the path
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x_t = conditional_path(x_0, x_1, t_val)
            expected = (1 - t_val) * x_0 + t_val * x_1
            assert torch.allclose(x_t, expected, atol=1e-6), f"Failed at t={t_val}"


class TestConditionalVectorField:
    """Test the conditional_vector_field function"""
    
    def test_affine_velocity(self, conditional_vector_field):
        """Test that velocity equals x_1 - x_0 for affine paths"""
        x_0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        t = torch.tensor([[0.5], [0.7]])
        
        v = conditional_vector_field(x_0, x_1, t)
        expected = x_1 - x_0
        
        assert torch.allclose(v, expected, atol=1e-6), "Velocity should be x_1 - x_0"
    
    def test_constant_in_time(self, conditional_vector_field):
        """Test that velocity is constant for affine paths (independent of t)"""
        x_0 = torch.tensor([[0.0, 0.0]])
        x_1 = torch.tensor([[10.0, 20.0]])
        
        # Velocity should be the same at different times
        v_0 = conditional_vector_field(x_0, x_1, 0.0)
        v_mid = conditional_vector_field(x_0, x_1, 0.5)
        v_1 = conditional_vector_field(x_0, x_1, 1.0)
        
        assert torch.allclose(v_0, v_mid, atol=1e-6), "Velocity should be constant across time"
        assert torch.allclose(v_mid, v_1, atol=1e-6), "Velocity should be constant across time"
    
    def test_batch_processing(self, conditional_vector_field):
        """Test batch processing"""
        batch_size = 50
        x_0 = torch.randn(batch_size, 2)
        x_1 = torch.randn(batch_size, 2)
        t = torch.rand(batch_size, 1)
        
        v = conditional_vector_field(x_0, x_1, t)
        
        assert v.shape == (batch_size, 2), "Output shape should match input"
        assert torch.allclose(v, x_1 - x_0, atol=1e-5), "All velocities should be x_1 - x_0"


class TestVectorFieldToScore:
    """Test the vector_field_to_score function"""
    
    def test_formula(self, vector_field_to_score):
        """Test that score function formula is correct: s(t,x) = (t*u - x) / (1-t)"""
        # Create a mock model that returns a known vector field
        class MockModel:
            def __call__(self, x, t):
                # Return a simple linear function for testing
                return 2 * x + t
        
        model = MockModel()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.tensor([[0.3], [0.5]])
        
        score = vector_field_to_score(model, t, x)
        
        # Compute expected score
        u = model(x, t)
        expected = (t * u - x) / (1 - t + 1e-5)
        
        assert torch.allclose(score, expected, atol=1e-5), "Score formula incorrect"
    
    def test_numerical_stability(self, vector_field_to_score):
        """Test that function handles t close to 1 without division by zero"""
        class MockModel:
            def __call__(self, x, t):
                return torch.ones_like(x)
        
        model = MockModel()
        x = torch.tensor([[1.0, 2.0]])
        t = torch.tensor([[0.999]])  # Very close to 1
        
        score = vector_field_to_score(model, t, x)
        
        # Should not produce NaN or Inf
        assert torch.all(torch.isfinite(score)), "Score should be finite even for t≈1"
    
    def test_shape_preservation(self, vector_field_to_score):
        """Test that output shape matches input shape"""
        class MockModel:
            def __call__(self, x, t):
                return torch.zeros_like(x)
        
        model = MockModel()
        batch_sizes = [1, 10, 100]
        dim = 6
        
        for bs in batch_sizes:
            x = torch.randn(bs, dim)
            t = torch.rand(bs, 1)
            score = vector_field_to_score(model, t, x)
            assert score.shape == (bs, dim), f"Shape mismatch for batch_size={bs}"


class TestSampleWithFlow:
    """Test the sample_with_flow function"""
    
    def test_output_shape(self, sample_with_flow):
        """Test that output has correct shape"""
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return torch.zeros_like(x)
        
        model = SimpleModel()
        n_samples = 20
        n_steps = 10
        
        trajectory = sample_with_flow(model, n_samples, n_steps)
        
        expected_shape = (n_steps + 1, n_samples, 2)  # Including initial point
        assert trajectory.shape == expected_shape, f"Expected shape {expected_shape}, got {trajectory.shape}"
    
    def test_deterministic(self, sample_with_flow):
        """Test that flow sampling is deterministic (same seed gives same output)"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return 0.1 * torch.ones_like(x)
        
        model = SimpleModel()
        
        # Run twice with same seed
        torch.manual_seed(42)
        traj1 = sample_with_flow(model, 10, 20)
        
        torch.manual_seed(42)
        traj2 = sample_with_flow(model, 10, 20)
        
        assert np.allclose(traj1, traj2, atol=1e-6), "Flow sampling should be deterministic"
    
    def test_initial_condition(self, sample_with_flow):
        """Test that initial samples come from source distribution"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return torch.zeros_like(x)
        
        model = SimpleModel()
        torch.manual_seed(42)
        trajectory = sample_with_flow(model, 1000, 10)
        
        # First timestep should be approximately N(0, 1)
        initial = trajectory[0]
        mean = np.mean(initial, axis=0)
        std = np.std(initial, axis=0)
        
        assert np.allclose(mean, 0, atol=0.2), "Initial mean should be close to 0"
        # assert np.allclose(std, 1, atol=0.2), "Initial std should be close to 1"


class TestSampleWithDiffusion:
    """Test the sample_with_diffusion function"""
    
    def test_output_shape(self, sample_with_diffusion):
        """Test that output has correct shape"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return torch.zeros_like(x)
        
        model = SimpleModel()
        n_samples = 20
        n_steps = 10
        
        trajectory = sample_with_diffusion(model, n_samples, n_steps, sigma=0.5)
        
        expected_shape = (n_steps + 1, n_samples, 2)
        assert trajectory.shape == expected_shape, f"Expected shape {expected_shape}, got {trajectory.shape}"
    
    def test_stochastic(self, sample_with_diffusion):
        """Test that diffusion sampling is stochastic (different runs give different results)"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return 0.1 * torch.ones_like(x)
        
        model = SimpleModel()
        
        # Run twice with different seeds
        torch.manual_seed(42)
        traj1 = sample_with_diffusion(model, 10, 20, sigma=0.5)
        
        torch.manual_seed(123)
        traj2 = sample_with_diffusion(model, 10, 20, sigma=0.5)
        
        # Should be different (with high probability)
        assert not np.allclose(traj1, traj2, atol=1e-3), "Diffusion sampling should be stochastic"
    
    def test_sigma_zero_equals_flow(self, sample_with_diffusion, sample_with_flow):
        """Test that diffusion with sigma=0 approximates deterministic flow"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return 0.5 * torch.ones_like(x)
        
        model = SimpleModel()
        
        torch.manual_seed(42)
        traj_flow = sample_with_flow(model, 50, 30)
        
        torch.manual_seed(42)
        traj_diff = sample_with_diffusion(model, 50, 30, sigma=0.0)
        
        # Should be very similar (but not exactly due to score computation)
        assert np.allclose(traj_flow, traj_diff, atol=1e-2), "Diffusion with σ=0 should approximate flow"
    
    def test_increasing_sigma_increases_variance(self, sample_with_diffusion):
        """Test that larger sigma leads to more stochastic trajectories"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = 'test'
            
            def forward(self, x, t):
                return torch.zeros_like(x)
        
        model = SimpleModel()
        n_samples = 100
        
        torch.manual_seed(42)
        traj_low = sample_with_diffusion(model, n_samples, 50, sigma=0.1)
        
        torch.manual_seed(42)
        traj_high = sample_with_diffusion(model, n_samples, 50, sigma=0.8)
        
        # Compute path variance for each trajectory
        var_low = np.var(traj_low[1:] - traj_low[:-1])  # Variance of steps
        var_high = np.var(traj_high[1:] - traj_high[:-1])
        
        assert var_high > var_low, "Higher sigma should produce more variable paths"


# ============================================================
# Test Runner with Clear Feedback
# ============================================================

def run_tests(conditional_path=None, 
              conditional_vector_field=None, 
              vector_field_to_score=None,
              sample_with_flow=None,
              sample_with_diffusion=None):
    """
    Run all tests with student implementations
    
    Args:
        conditional_path: Student's implementation of conditional path
        conditional_vector_field: Student's implementation of vector field
        vector_field_to_score: Student's implementation of score conversion
        sample_with_flow: Student's implementation of flow sampling
        sample_with_diffusion: Student's implementation of diffusion sampling
    
    Returns:
        Dictionary with test results
    """
    results = {
        'conditional_path': None,
        'conditional_vector_field': None,
        'vector_field_to_score': None,
        'sample_with_flow': None,
        'sample_with_diffusion': None
    }
    
    print("="*70)
    print("RUNNING FLOW MATCHING EXERCISE TESTS")
    print("="*70)
    print()
    
    # Test 1: Conditional Path
    if conditional_path is not None:
        print("📝 Testing conditional_path()...")
        test_class = TestConditionalPath()
        try:
            test_class.test_boundary_conditions(conditional_path)
            print("  ✓ Boundary conditions test passed")
            test_class.test_midpoint(conditional_path)
            print("  ✓ Midpoint test passed")
            test_class.test_batch_processing(conditional_path)
            print("  ✓ Batch processing test passed")
            test_class.test_linearity(conditional_path)
            print("  ✓ Linearity test passed")
            results['conditional_path'] = 'PASSED'
            print("  ✅ All conditional_path tests PASSED!\n")
        except AssertionError as e:
            results['conditional_path'] = 'FAILED'
            print(f"  ❌ FAILED: {e}\n")
        except Exception as e:
            results['conditional_path'] = 'ERROR'
            print(f"  ⚠️  ERROR: {e}\n")
    else:
        print("⏭️  Skipping conditional_path (not provided)\n")
    
    # Test 2: Conditional Vector Field
    if conditional_vector_field is not None:
        print("📝 Testing conditional_vector_field()...")
        test_class = TestConditionalVectorField()
        try:
            test_class.test_affine_velocity(conditional_vector_field)
            print("  ✓ Affine velocity test passed")
            test_class.test_constant_in_time(conditional_vector_field)
            print("  ✓ Time independence test passed")
            test_class.test_batch_processing(conditional_vector_field)
            print("  ✓ Batch processing test passed")
            results['conditional_vector_field'] = 'PASSED'
            print("  ✅ All conditional_vector_field tests PASSED!\n")
        except AssertionError as e:
            results['conditional_vector_field'] = 'FAILED'
            print(f"  ❌ FAILED: {e}\n")
        except Exception as e:
            results['conditional_vector_field'] = 'ERROR'
            print(f"  ⚠️  ERROR: {e}\n")
    else:
        print("⏭️  Skipping conditional_vector_field (not provided)\n")
    
    # Test 3: Vector Field to Score
    if vector_field_to_score is not None:
        print("📝 Testing vector_field_to_score()...")
        test_class = TestVectorFieldToScore()
        try:
            test_class.test_formula(vector_field_to_score)
            print("  ✓ Formula test passed")
            test_class.test_numerical_stability(vector_field_to_score)
            print("  ✓ Numerical stability test passed")
            test_class.test_shape_preservation(vector_field_to_score)
            print("  ✓ Shape preservation test passed")
            results['vector_field_to_score'] = 'PASSED'
            print("  ✅ All vector_field_to_score tests PASSED!\n")
        except AssertionError as e:
            results['vector_field_to_score'] = 'FAILED'
            print(f"  ❌ FAILED: {e}\n")
        except Exception as e:
            results['vector_field_to_score'] = 'ERROR'
            print(f"  ⚠️  ERROR: {e}\n")
    else:
        print("⏭️  Skipping vector_field_to_score (not provided)\n")
    
    # Test 4: Sample with Flow
    if sample_with_flow is not None:
        print("📝 Testing sample_with_flow()...")
        test_class = TestSampleWithFlow()
        try:
            test_class.test_output_shape(sample_with_flow)
            print("  ✓ Output shape test passed")
            test_class.test_deterministic(sample_with_flow)
            print("  ✓ Deterministic test passed")
            test_class.test_initial_condition(sample_with_flow)
            print("  ✓ Initial condition test passed")
            results['sample_with_flow'] = 'PASSED'
            print("  ✅ All sample_with_flow tests PASSED!\n")
        except AssertionError as e:
            results['sample_with_flow'] = 'FAILED'
            print(f"  ❌ FAILED: {e}\n")
        except Exception as e:
            results['sample_with_flow'] = 'ERROR'
            print(f"  ⚠️  ERROR: {e}\n")
    else:
        print("⏭️  Skipping sample_with_flow (not provided)\n")
    
    # Test 5: Sample with Diffusion
    if sample_with_diffusion is not None:
        print("📝 Testing sample_with_diffusion()...")
        test_class = TestSampleWithDiffusion()
        try:
            test_class.test_output_shape(sample_with_diffusion)
            print("  ✓ Output shape test passed")
            test_class.test_stochastic(sample_with_diffusion)
            print("  ✓ Stochastic test passed")
            # Note: The sigma_zero test requires sample_with_flow too
            if sample_with_flow is not None:
                test_class.test_sigma_zero_equals_flow(sample_with_diffusion, sample_with_flow)
                print("  ✓ Sigma=0 approximates flow test passed")
            test_class.test_increasing_sigma_increases_variance(sample_with_diffusion)
            print("  ✓ Sigma effect test passed")
            results['sample_with_diffusion'] = 'PASSED'
            print("  ✅ All sample_with_diffusion tests PASSED!\n")
        except AssertionError as e:
            results['sample_with_diffusion'] = 'FAILED'
            print(f"  ❌ FAILED: {e}\n")
        except Exception as e:
            results['sample_with_diffusion'] = 'ERROR'
            print(f"  ⚠️  ERROR: {e}\n")
    else:
        print("⏭️  Skipping sample_with_diffusion (not provided)\n")
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v == 'PASSED')
    failed = sum(1 for v in results.values() if v == 'FAILED')
    errors = sum(1 for v in results.values() if v == 'ERROR')
    skipped = sum(1 for v in results.values() if v is None)
    
    for func_name, status in results.items():
        if status == 'PASSED':
            print(f"  ✅ {func_name}: PASSED")
        elif status == 'FAILED':
            print(f"  ❌ {func_name}: FAILED")
        elif status == 'ERROR':
            print(f"  ⚠️  {func_name}: ERROR")
        else:
            print(f"  ⏭️  {func_name}: SKIPPED")
    
    print()
    print(f"Total: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
    print("="*70)
    
    return results


if __name__ == "__main__":
    print("This test suite should be imported and used from the notebook.")
    print("Example usage:")
    print("  from test_exercises import run_tests")
    print("  run_tests(conditional_path, conditional_vector_field, ...)")
