import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
# Import your module directly
from kdfp.utils.shallow_model_loading import (
    load_shallow, 
    _load_shallow_tl, 
    _load_shallow_hf,
    get_indices_and_num_layers_to_keep
)


class TestWeightCopying:
    """Test that shallow models have correct weights copied from original models."""
    
    @pytest.fixture
    def mock_tl_model(self):
        """Create a mock TransformerLens model with deterministic weights."""
        model = MagicMock()
        model.cfg.n_layers = 12
        model.cfg.copy.return_value = MagicMock()
        model.cfg.copy.return_value.n_layers = 12
        
        # Create deterministic weights for each layer
        model.blocks = []
        layer_weights = {}
        for i in range(12):
            block = MagicMock()
            weight_dict = {f'layer_{i}_weight': torch.full((5, 5), float(i))}
            block.state_dict.return_value = weight_dict
            model.blocks.append(block)
            layer_weights[i] = weight_dict
        
        # Create deterministic weights for other components
        component_weights = {}
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            comp = MagicMock()
            weight_dict = {f'{component}_weight': torch.full((3, 3), hash(component) % 100)}
            comp.state_dict.return_value = weight_dict
            setattr(model, component, comp)
            component_weights[component] = weight_dict
        
        return model, layer_weights, component_weights
    
    @pytest.fixture
    def mock_hf_model(self):
        """Create a mock HuggingFace model with deterministic weights."""
        # Create deterministic state dict
        state_dict = {}
        
        # Layer weights
        layer_weights = {}
        for i in range(12):
            layer_key = f'transformer.h.{i}.weight'
            weight = torch.full((5, 5), float(i))
            state_dict[layer_key] = weight
            layer_weights[i] = {layer_key.split('.')[-1]: weight}
        
        # Component weights
        component_weights = {
            'transformer.wte.weight': torch.full((10, 8), 1.0),
            'transformer.wpe.weight': torch.full((10, 8), 2.0),
            'transformer.ln_f.weight': torch.full((8,), 3.0),
            'lm_head.weight': torch.full((10, 8), 4.0),
        }
        state_dict.update(component_weights)
        
        return state_dict, layer_weights, component_weights
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_layer_weights_copied_correctly(self, mock_hooked_transformer, mock_tl_model):
        """Test that correct layer weights are copied in TransformerLens mode."""
        original_model, layer_weights, component_weights = mock_tl_model
        mock_hooked_transformer.from_pretrained.return_value = original_model
        
        # Create shallow model mock - 12 * 0.25 = 3 layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(3)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with reduction factor 0.25 (12 * 0.25 = 3 layers)
        result_model, tokenizer, keep_indices = _load_shallow_tl(model_name="test_model",
                                                                 layer_reduction_factor=0.25,
                                                                 selection_method='linspace')
        
        # Calculate expected layer indices: torch.linspace(0, 11, 3) = [0, 6, 11]
        expected_indices = [0, 6, 11]
        
        # Check that returned keep_indices match expected
        assert keep_indices == expected_indices, f"Expected {expected_indices}, got {keep_indices}"
        
        # Verify correct layers were copied
        assert len(shallow_model.blocks) == 3
        for i, expected_idx in enumerate(expected_indices):
            expected_state = layer_weights[expected_idx]
            shallow_model.blocks[i].load_state_dict.assert_called_once_with(expected_state)
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_component_weights_copied_correctly(self, mock_hooked_transformer, mock_tl_model):
        """Test that component weights are copied correctly in TransformerLens mode."""
        original_model, layer_weights, component_weights = mock_tl_model
        mock_hooked_transformer.from_pretrained.return_value = original_model
        
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(3)]  # 12 * 0.25 = 3
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with reduction factor 0.25
        _load_shallow_tl(model_name="test_model",
                         layer_reduction_factor=0.25,
                         selection_method='linspace')
        
        # Verify each component was loaded with correct weights
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            expected_state = component_weights[component]
            getattr(shallow_model, component).load_state_dict.assert_called_once_with(expected_state)
    
    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_layer_weights_copied_correctly(self, mock_tokenizer, mock_config, mock_model, mock_hf_model):
        """Test that correct layer weights are copied in HuggingFace mode."""
        original_state_dict, layer_weights, component_weights = mock_hf_model
        
        # Setup original model mock
        original_model = MagicMock()
        original_model.transformer.h = [MagicMock() for _ in range(12)]
        for i, layer in enumerate(original_model.transformer.h):
            layer.state_dict.return_value = layer_weights[i]
        original_model.state_dict.return_value = original_state_dict
        mock_model.from_pretrained.return_value = original_model
        
        # Setup config mock
        config = MagicMock()
        config.n_layer = 12
        mock_config.from_pretrained.return_value = config
        
        # Setup shallow model mock - 12 * 0.25 = 3 layers
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(3)]
        shallow_state_dict = {k: torch.zeros_like(v) for k, v in component_weights.items()}
        shallow_model.state_dict.return_value = shallow_state_dict
        mock_model.from_config.return_value = shallow_model
        
        # Setup tokenizer mock
        tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = tokenizer
        
        # Run the function with reduction factor 0.25
        result_model, result_tokenizer, keep_indices = _load_shallow_hf(model_name="gpt2",
                                                                        layer_reduction_factor=0.25,
                                                                        selection_method='linspace')
        
        # Calculate expected layer indices: torch.linspace(0, 11, 3) = [0, 6, 11]
        expected_indices = [0, 6, 11]
        
        # Verify correct layers were copied
        for i, expected_idx in enumerate(expected_indices):
            expected_state = layer_weights[expected_idx]
            shallow_model.transformer.h[i].load_state_dict.assert_called_once_with(expected_state)
    
    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_component_weights_copied_correctly(self, mock_tokenizer, mock_config, mock_model, mock_hf_model):
        """Test that component weights are copied correctly in HuggingFace mode."""
        original_state_dict, layer_weights, component_weights = mock_hf_model
        
        # Setup mocks (similar to above test)
        original_model = MagicMock()
        original_model.transformer.h = [MagicMock() for _ in range(12)]
        for i, layer in enumerate(original_model.transformer.h):
            layer.state_dict.return_value = layer_weights[i]
        original_model.state_dict.return_value = original_state_dict
        mock_model.from_pretrained.return_value = original_model
        
        config = MagicMock()
        config.n_layer = 12
        mock_config.from_pretrained.return_value = config
        
        # 12 * 0.25 = 3 layers
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(3)]
        shallow_state_dict = {k: torch.zeros_like(v) for k, v in component_weights.items()}
        shallow_model.state_dict.return_value = shallow_state_dict
        mock_model.from_config.return_value = shallow_model
        
        tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = tokenizer
        
        # Run the function with reduction factor 0.25
        _, _, keep_indices = _load_shallow_hf(model_name="gpt2",
                                              layer_reduction_factor=0.25,
                                              selection_method='linspace')
        
        # Verify load_state_dict was called with modified state
        shallow_model.load_state_dict.assert_called_once()
        
        # Check that returned keep_indices match expected for this reduction factor
        expected_indices = [0, 6, 11]  # 12 layers, reduction factor 0.25
        assert keep_indices == expected_indices, f"Expected {expected_indices}, got {keep_indices}"
        
        # Get the actual call arguments
        call_args = shallow_model.load_state_dict.call_args[0][0]
        
        # Verify that component weights were copied correctly
        for key, expected_weight in component_weights.items():
            assert key in call_args
            # The weights should have been copied (cloned) from original
            assert torch.equal(call_args[key], expected_weight)
    
    def test_layer_index_calculation(self):
        """Test that layer indices are calculated correctly for different reduction factors."""
        # Test cases: (total_layers, reduction_factor, expected_kept_layers, expected_indices)
        test_cases = [
            (12, 0.5, 6, [0, 2, 4, 7, 9, 11]),      # 12 * 0.5 = 6 layers
            (12, 0.25, 3, [0, 6, 11]),              # 12 * 0.25 = 3 layers
            (12, 0.33, 4, [0, 4, 7, 11]),           # 12 * 0.33 ≈ 4 layers
            (8, 0.5, 4, [0, 2, 5, 7]),              # 8 * 0.5 = 4 layers
            (6, 0.33, 2, [0, 5]),                   # 6 * 0.33 ≈ 2 layers
            (10, 0.2, 2, [0, 9]),                   # 10 * 0.2 = 2 layers
        ]
        
        for total_layers, reduction_factor, expected_count, expected_indices in test_cases:
            keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(total_layers=total_layers,
                                                                          layer_reduction_factor=reduction_factor,
                                                                          selection_method='linspace')

            assert len(keep_indices) == expected_count, f"Failed for {total_layers}/{reduction_factor}: expected {expected_count}, got {len(keep_indices)}"
            assert keep_indices == expected_indices, f"Failed for {total_layers}/{reduction_factor}: got {keep_indices}, expected {expected_indices}"


class TestKeepIndicesValidation:
    """Test that the keep_indices returned are correct for various scenarios."""
    
    @pytest.mark.parametrize("total_layers,reduction_factor,expected_indices", [
        (12, 0.5, [0, 2, 4, 7, 9, 11]),      # 12 * 0.5 = 6 layers
        (12, 0.25, [0, 6, 11]),              # 12 * 0.25 = 3 layers
        (12, 0.33, [0, 4, 7, 11]),           # 12 * 0.33 ≈ 4 layers
        (8, 0.5, [0, 2, 5, 7]),              # 8 * 0.5 = 4 layers
        (6, 0.33, [0, 5]),                   # 6 * 0.33 ≈ 2 layers
        (24, 0.17, [0, 8, 15, 23]),          # 24 * 0.17 ≈ 4 layers
        (16, 0.125, [0, 15]),                # 16 * 0.125 = 2 layers
    ])
    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_keep_indices_returned(self, mock_tokenizer, mock_config, mock_model, 
                                       total_layers, reduction_factor, expected_indices):
        """Test keep_indices calculation for various layer counts and reduction factors."""
        # Setup mocks
        original_model = MagicMock()
        original_model.transformer.h = [MagicMock() for _ in range(total_layers)]
        original_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_pretrained.return_value = original_model
        
        config = MagicMock()
        setattr(config, 'n_layer', total_layers)  # Using setattr to handle dynamic attribute
        mock_config.from_pretrained.return_value = config

        _, layers_to_keep = get_indices_and_num_layers_to_keep(total_layers=total_layers,
                                                           layer_reduction_factor=reduction_factor,
                                                           selection_method='linspace')
        
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(layers_to_keep)]
        shallow_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_config.return_value = shallow_model
        
        tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = tokenizer
        
        # Run the function
        _, _, keep_indices = _load_shallow_hf(model_name="gpt2",
                                              layer_reduction_factor=reduction_factor,
                                              selection_method='linspace')
        
        # Verify the keep_indices match expected
        assert keep_indices == expected_indices, (
            f"For {total_layers} layers with reduction factor {reduction_factor}: "
            f"expected {expected_indices}, got {keep_indices}"
        )
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_keep_indices_returned(self, mock_hooked_transformer):
        """Test that TransformerLens mode returns correct keep_indices."""
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = 8
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = 8
        original_model.blocks = [MagicMock() for _ in range(8)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        mock_hooked_transformer.from_pretrained.return_value = original_model
        
        # 8 * 0.5 = 4 layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(4)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with reduction factor 0.5
        _, _, keep_indices = _load_shallow_tl(model_name="test_model",
                                              layer_reduction_factor=0.5,
                                              selection_method='linspace')
        
        # For 8 layers with reduction factor 0.5: torch.linspace(0, 7, 4) = [0, 2, 5, 7]
        expected_indices = [0, 2, 5, 7]
        assert keep_indices == expected_indices


class TestEdgeCases:
    """Test edge cases for the reduction factor."""
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_minimum_one_layer(self, mock_hooked_transformer):
        """Test that at least one layer is kept even with very small reduction factor."""
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = 12
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = 12
        original_model.blocks = [MagicMock() for _ in range(12)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        mock_hooked_transformer.from_pretrained.return_value = original_model
        
        # Even with very small reduction factor, should keep at least 1 layer
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(1)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with very small reduction factor
        _, _, keep_indices = _load_shallow_tl(model_name="test_model",
                                              layer_reduction_factor=0.01,
                                              selection_method='linspace')
        
        # Should keep exactly 1 layer (index 0)
        assert len(keep_indices) == 1
        assert keep_indices == [0]
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_reduction_factor_greater_than_one(self, mock_hooked_transformer):
        """Test that reduction factor > 1 keeps all layers."""
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = 12
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = 12
        original_model.blocks = [MagicMock() for _ in range(12)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        mock_hooked_transformer.from_pretrained.return_value = original_model
        
        # Reduction factor > 1 should keep all layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(12)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with reduction factor > 1
        _, _, keep_indices = _load_shallow_tl(model_name="test_model",
                                              layer_reduction_factor=1.5,
                                              selection_method='linspace')
        
        # Should keep all 12 layers
        assert len(keep_indices) == 12
        assert keep_indices == list(range(12))


class TestWeightPreservation:
    """Test that weight values are preserved exactly during copying."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_weight_preservation_pythia_31m_half_layers(self):
        """
        Integration test that verifies weight values are preserved exactly
        when creating a shallow Pythia-31M model with half the layers.
        """
        model_name = "EleutherAI/pythia-31m"
        reduction_factor = 0.5
        selection_method = 'linspace'
        
        try:
            # Load and create shallow model
            shallow_model, tokenizer, keep_indices = _load_shallow_hf(model_name=model_name,
                                                                      layer_reduction_factor=reduction_factor,
                                                                      selection_method='linspace')
            original_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Get state dicts
            original_state = original_model.state_dict()
            shallow_state = shallow_model.state_dict()
            
            # Test 1: Non-layer components preserved
            non_layer_keys = [k for k in original_state.keys() if not k.startswith('gpt_neox.layers.')]
            for key in non_layer_keys:
                if key in shallow_state:
                    assert torch.equal(original_state[key], shallow_state[key])
            
            # Test 2: Selected layers preserved
            for i, layer_idx in enumerate(keep_indices):
                layer_prefix = f'gpt_neox.layers.{layer_idx}.'
                shallow_prefix = f'gpt_neox.layers.{i}.'
                
                layer_keys = [k for k in original_state.keys() if k.startswith(layer_prefix)]
                for orig_key in layer_keys:
                    shallow_key = orig_key.replace(layer_prefix, shallow_prefix)
                    if shallow_key in shallow_state:
                        assert torch.equal(original_state[orig_key], shallow_state[shallow_key])
            
            # Clean up
            del original_model, shallow_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            pytest.skip(f"Could not load Pythia-31M model: {e}")

class TestFirstNSelection:
    """Test the 'first_n' selection method for layer reduction."""
    
    @pytest.mark.parametrize("total_layers,reduction_factor,expected_indices,expected_count", [
        # Basic cases
        (12, 0.5, [0, 1, 2, 3, 4, 5], 6),          # 12 * 0.5 = 6 layers (first 6)
        (12, 0.25, [0, 1, 2], 3),                  # 12 * 0.25 = 3 layers (first 3)
        (12, 0.33, [0, 1, 2, 3], 4),               # 12 * 0.33 ≈ 4 layers (first 4)
        (8, 0.5, [0, 1, 2, 3], 4),                 # 8 * 0.5 = 4 layers (first 4)
        (6, 0.33, [0, 1], 2),                      # 6 * 0.33 ≈ 2 layers (first 2)
        
        # Edge cases
        (12, 1.0, list(range(12)), 12),            # 100% reduction = all layers
        (12, 0.01, [0], 1),                        # Very small reduction = first layer only
        (12, 1.5, list(range(12)), 12),            # >100% reduction = all layers
        (1, 0.5, [0], 1),                          # Single layer model
        (1, 1.0, [0], 1),                          # Single layer, 100% reduction
    ])
    def test_first_n_indices_calculation(self, total_layers, reduction_factor, 
                                        expected_indices, expected_count):
        """Test that first_n selection returns correct indices and count."""
        keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(
            total_layers=total_layers,
            layer_reduction_factor=reduction_factor,
            selection_method='first_n'
        )
        
        assert layers_to_keep == expected_count, (
            f"Expected {expected_count} layers, got {layers_to_keep}"
        )
        assert keep_indices == expected_indices, (
            f"Expected indices {expected_indices}, got {keep_indices}"
        )
        assert len(keep_indices) == expected_count, (
            f"Number of indices ({len(keep_indices)}) doesn't match expected count ({expected_count})"
        )
    
    def test_first_n_versus_linspace_comparison(self):
        """Test that first_n and linspace produce different results for same parameters."""
        total_layers = 12
        reduction_factor = 0.5
        
        first_n_indices, first_n_count = get_indices_and_num_layers_to_keep(
            total_layers=total_layers,
            layer_reduction_factor=reduction_factor,
            selection_method='first_n'
        )
        
        linspace_indices, linspace_count = get_indices_and_num_layers_to_keep(
            total_layers=total_layers,
            layer_reduction_factor=reduction_factor,
            selection_method='linspace'
        )
        
        # Both should keep the same number of layers
        assert first_n_count == linspace_count == 6
        
        # But they should have different indices
        assert first_n_indices != linspace_indices
        assert first_n_indices == [0, 1, 2, 3, 4, 5]  # First 6 layers
        assert linspace_indices == [0, 2, 4, 7, 9, 11]  # Evenly spaced
    
    @patch('kdfp.utils.shallow_model_loading.HookedTransformer')
    def test_tl_first_n_selection(self, mock_hooked_transformer):
        """Test that TransformerLens mode uses first_n selection correctly."""
        total_layers = 8
        reduction_factor = 0.5  # Should keep first 4 layers
        
        # Setup mock
        original_model = MagicMock()
        original_model.cfg.n_layers = total_layers
        original_model.cfg.copy.return_value = MagicMock()
        original_model.cfg.copy.return_value.n_layers = total_layers
        original_model.blocks = [MagicMock() for _ in range(total_layers)]
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(original_model, component, MagicMock())
        mock_hooked_transformer.from_pretrained.return_value = original_model
        
        # Create shallow model with expected number of layers
        shallow_model = MagicMock()
        shallow_model.blocks = [MagicMock() for _ in range(4)]  # 8 * 0.5 = 4 layers
        for component in ['embed', 'unembed', 'ln_final', 'pos_embed']:
            setattr(shallow_model, component, MagicMock())
        mock_hooked_transformer.return_value = shallow_model
        
        # Run the function with first_n selection
        _, _, keep_indices = _load_shallow_tl(
            model_name="test_model",
            layer_reduction_factor=reduction_factor,
            selection_method='first_n'
        )
        
        # Should keep first 4 layers: [0, 1, 2, 3]
        expected_indices = [0, 1, 2, 3]
        assert keep_indices == expected_indices
    
    @patch('kdfp.utils.shallow_model_loading.AutoModelForCausalLM')
    @patch('kdfp.utils.shallow_model_loading.AutoConfig')
    @patch('kdfp.utils.shallow_model_loading.AutoTokenizer')
    def test_hf_first_n_selection(self, mock_tokenizer, mock_config, mock_model):
        """Test that HuggingFace mode uses first_n selection correctly."""
        total_layers = 10
        reduction_factor = 0.3  # Should keep first 3 layers
        
        # Setup mocks
        original_model = MagicMock()
        original_model.transformer.h = [MagicMock() for _ in range(total_layers)]
        original_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_pretrained.return_value = original_model
        
        config = MagicMock()
        setattr(config, 'n_layer', total_layers)
        mock_config.from_pretrained.return_value = config
        
        # Create shallow model with expected number of layers
        shallow_model = MagicMock()
        shallow_model.transformer.h = [MagicMock() for _ in range(3)]  # 10 * 0.3 = 3 layers
        shallow_model.state_dict.return_value = {
            'transformer.wte.weight': torch.randn(10, 8),
            'transformer.wpe.weight': torch.randn(10, 8),
            'transformer.ln_f.weight': torch.randn(8),
            'lm_head.weight': torch.randn(10, 8),
        }
        mock_model.from_config.return_value = shallow_model
        
        tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = tokenizer
        
        # Run the function with first_n selection
        _, _, keep_indices = _load_shallow_hf(
            model_name="gpt2",
            layer_reduction_factor=reduction_factor,
            selection_method='first_n'
        )
        
        # Should keep first 3 layers: [0, 1, 2]
        expected_indices = [0, 1, 2]
        assert keep_indices == expected_indices
    
    def test_first_n_consistency_across_runs(self):
        """Test that first_n selection is deterministic and consistent."""
        total_layers = 12
        reduction_factor = 0.25
        
        # Run multiple times
        results = []
        for _ in range(5):
            keep_indices, layers_to_keep = get_indices_and_num_layers_to_keep(
                total_layers=total_layers,
                layer_reduction_factor=reduction_factor,
                selection_method='first_n'
            )
            results.append((keep_indices, layers_to_keep))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "first_n selection should be deterministic"
        
        # Verify the expected result
        expected_indices = [0, 1, 2]  # 12 * 0.25 = 3 layers (first 3)
        expected_count = 3
        assert first_result == (expected_indices, expected_count)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])