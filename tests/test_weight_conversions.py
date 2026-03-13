import pytest
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from kdfp.utils.weight_conversions.pythia_wc import (
    convert_hooked_to_neox_weights,
    convert_hooked_to_gpt2_weights,
    convert_gpt2_to_hooked_weights,
)
import einops

class TestRealModelConversion:
    """Test that converted weights match original HuggingFace models."""
    def _test_conversion_matching(self, model_name: str, conversion_fn):
        try:
            
            print("Loading original HuggingFace model...")
            # Load original HF model for comparison
            hf_original = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)  # Force float32
            hf_original_state = hf_original.state_dict()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print("Loading HookedTransformer and converting weights...")
            # Load HookedTransformer and convert to HF format
            hooked_model = HookedTransformer.from_pretrained_no_processing(model_name)
            hooked_state = hooked_model.state_dict()
            
            # Convert to HF format using your function
            converted_state, missing = conversion_fn(hooked_state, hooked_model.cfg)
            assert len(missing) == 0, f"Missing params: {missing}"
            
            print("Creating HF model with converted weights...")
            # Create new HF model with converted weights (also in float32)
            hf_config = AutoConfig.from_pretrained(model_name)
            hf_converted = AutoModelForCausalLM.from_config(hf_config).float()  # Convert to float32
            hf_converted.load_state_dict(converted_state, strict=True)
            hf_converted_state = hf_converted.state_dict()
            
            print("Comparing weights between original and converted HF models...")
            # Compare weights directly
            self._compare_state_dicts(hf_original_state, hf_converted_state, tol=1e-6)
            
            print("Testing forward pass consistency...")
            # Test that they produce the same outputs
            test_text = "Hello, how are you today?"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                original_output = hf_original(**inputs)
                converted_output = hf_converted(**inputs)
            
            # Convert both to same dtype for comparison
            original_logits = original_output.logits.float()
            converted_logits = converted_output.logits.float()
            
            # Check logits are similar
            assert torch.allclose(original_logits, converted_logits, atol=1e-5, rtol=1e-3)
            
            print(f"✅ {model_name.upper()} conversion matches original HF model!")
            
        except Exception as e:
            pytest.fail(f"{model_name.upper()} test failed: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pythia_31m_conversion_matches_hf(self):
        """Test that Pythia-31M converted weights match original HF weights."""
        model_name = "EleutherAI/pythia-31m"
        conversion_fn = convert_hooked_to_neox_weights
        self._test_conversion_matching(model_name, conversion_fn)

    # @pytest.mark.integration
    # @pytest.mark.slow
    # def test_gpt2_conversion_matches_hf(self):
    #     """Test that Pythia-31M converted weights match original HF weights."""
    #     model_name = "gpt2"
    #     conversion_fn = convert_hooked_to_gpt2_weights
    #     self._test_conversion_matching(model_name, conversion_fn)
            
    
    def _compare_state_dicts(self, dict1, dict2, tol=1e-6):
        """Compare two state dictionaries with tolerance."""
        # Check for key differences first
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        if keys1 != keys2:
            print(f"Key mismatch: {keys1 - keys2} vs {keys2 - keys1}")
            return False
        
        all_close = True
        max_diff = 0
        problematic_keys = []
        
        for key in dict1.keys():
            if key not in dict2:
                print(f"Key {key} missing in second dict")
                all_close = False
                continue
                
            if dict1[key].shape != dict2[key].shape:
                print(f"Shape mismatch for {key}: {dict1[key].shape} vs {dict2[key].shape}")
                all_close = False
                continue
            
            # Ensure both tensors are the same dtype for comparison
            tensor1 = dict1[key].float() if dict1[key].dtype != dict2[key].dtype else dict1[key]
            tensor2 = dict2[key].float() if dict1[key].dtype != dict2[key].dtype else dict2[key]
            
            diff = torch.max(torch.abs(tensor1 - tensor2)).item()
            if diff > tol:
                print(f"Warning: {key} has max difference {diff:.2e} (tolerance {tol:.1e})")
                all_close = False
                problematic_keys.append((key, diff))
                max_diff = max(max_diff, diff)
        
        if not all_close:
            print(f"Maximum difference found: {max_diff:.2e}")
            print("Top 5 problematic parameters:")
            for key, diff in sorted(problematic_keys, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {key}: {diff:.2e}")
        
        return all_close


#     def debug_conversion_roundtrip(self):
#         """
#         Debug the conversion by doing a full roundtrip test and checking intermediate steps
#         """
#         print("=== Loading Original Models ===")
        
#         # Load original HF model
#         hf_model = AutoModelForCausalLM.from_pretrained("gpt2").cpu()
#         tokenizer = AutoTokenizer.from_pretrained("gpt2")
#         tokenizer.pad_token = tokenizer.eos_token
        
#         # Load HookedTransformer version
#         hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2").cpu()
        
#         print("=== Step 1: Compare Original Models ===")
#         test_text = "The quick brown fox jumps over the"
#         tokens = tokenizer(test_text, return_tensors="pt")
        
#         with torch.no_grad():
#             hf_output = hf_model(**tokens).logits
#             hooked_output = hooked_model(tokens["input_ids"])
        
#         print(f"Original HF vs HookedTransformer allclose: {torch.allclose(hf_output, hooked_output, atol=1e-4)}")
#         print(f"Max diff: {torch.max(torch.abs(hf_output - hooked_output))}")
        
#         print("\n=== Step 2: Convert HF -> Hooked -> HF ===")
        
#         # Convert HF to Hooked format
#         hooked_state_dict = convert_gpt2_to_hooked_weights(hf_model, hooked_model.cfg)
        
#         # Convert back to HF format
#         converted_hf_state_dict, missing_params = convert_hooked_to_gpt2_weights(hooked_state_dict, hooked_model.cfg)
        
#         if missing_params:
#             print(f"Missing parameters: {missing_params}")
        
#         # Load converted weights into new model
#         converted_model = AutoModelForCausalLM.from_config(hf_model.config)
#         missing_keys, unexpected_keys = converted_model.load_state_dict(converted_hf_state_dict, strict=False)
        
#         if missing_keys:
#             print(f"Missing keys when loading: {missing_keys}")
#         if unexpected_keys:
#             print(f"Unexpected keys when loading: {unexpected_keys}")
        
#         print("\n=== Step 3: Compare Converted Model ===")
        
#         with torch.no_grad():
#             converted_output = converted_model(**tokens).logits
        
#         print(f"Original HF vs Converted allclose: {torch.allclose(hf_output, converted_output, atol=1e-4)}")
#         print(f"Max diff: {torch.max(torch.abs(hf_output - converted_output))}")
        
#         # More detailed comparison
#         print(f"Mean absolute error: {torch.mean(torch.abs(hf_output - converted_output))}")
#         print(f"Relative error: {torch.mean(torch.abs((hf_output - converted_output) / (hf_output + 1e-8)))}")
        
#         print("\n=== Step 4: Layer-by-Layer Weight Comparison ===")
        
#         original_state_dict = hf_model.state_dict()
        
#         # Compare key weights
#         weight_comparisons = [
#             "transformer.wte.weight",
#             "transformer.wpe.weight", 
#             "transformer.h.0.attn.c_attn.weight",
#             "transformer.h.0.attn.c_attn.bias",
#             "transformer.h.0.attn.c_proj.weight",
#             "transformer.h.0.mlp.c_fc.weight",
#             "transformer.h.0.mlp.c_proj.weight",
#             "lm_head.weight"
#         ]
        
#         for weight_name in weight_comparisons:
#             if weight_name in original_state_dict and weight_name in converted_hf_state_dict:
#                 orig = original_state_dict[weight_name]
#                 conv = converted_hf_state_dict[weight_name]
                
#                 shapes_match = orig.shape == conv.shape
#                 weights_close = torch.allclose(orig, conv, atol=1e-6)
#                 max_diff = torch.max(torch.abs(orig - conv))
                
#                 print(f"{weight_name}:")
#                 print(f"  Shapes match: {shapes_match} ({orig.shape} vs {conv.shape})")
#                 print(f"  Weights close: {weights_close}")
#                 print(f"  Max diff: {max_diff:.2e}")
#             else:
#                 print(f"{weight_name}: MISSING in one of the state dicts")
        
#         print("\n=== Step 5: Detailed Attention Weight Analysis ===")
        
#         # Let's specifically debug the attention weight conversion
#         layer_idx = 0
        
#         # Original HF attention weights
#         orig_qkv_weight = original_state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"]
#         orig_qkv_bias = original_state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"]
        
#         print(f"Original c_attn weight shape: {orig_qkv_weight.shape}")
#         print(f"Original c_attn bias shape: {orig_qkv_bias.shape}")
        
#         # Split like in the forward conversion
#         W_Q, W_K, W_V = torch.tensor_split(orig_qkv_weight, 3, dim=1)
#         print(f"Split Q, K, V shapes: {W_Q.shape}, {W_K.shape}, {W_V.shape}")
        
#         # Reshape like in forward conversion
#         n_heads = hooked_model.cfg.n_heads
#         W_Q_reshaped = einops.rearrange(W_Q, "m (i h)->i m h", i=n_heads)
#         W_K_reshaped = einops.rearrange(W_K, "m (i h)->i m h", i=n_heads)
#         W_V_reshaped = einops.rearrange(W_V, "m (i h)->i m h", i=n_heads)
        
#         print(f"Reshaped Q, K, V shapes: {W_Q_reshaped.shape}, {W_K_reshaped.shape}, {W_V_reshaped.shape}")
        
#         # Now reverse the process
#         W_Q_back = einops.rearrange(W_Q_reshaped, "i m h->m (i h)")
#         W_K_back = einops.rearrange(W_K_reshaped, "i m h->m (i h)")
#         W_V_back = einops.rearrange(W_V_reshaped, "i m h->m (i h)")
        
#         W_QKV_back = torch.cat([W_Q_back, W_K_back, W_V_back], dim=1)
        
#         print(f"Reconstructed QKV weight shape: {W_QKV_back.shape}")
#         print(f"QKV weight reconstruction perfect: {torch.allclose(orig_qkv_weight, W_QKV_back, atol=1e-8)}")
#         print(f"QKV weight max diff: {torch.max(torch.abs(orig_qkv_weight - W_QKV_back)):.2e}")
        
#         return hf_model, converted_model, hooked_model
    
#     @pytest.mark.integration
#     @pytest.mark.slow
#     def test_gpt2_conversion_roundtrip_debug(self):
#         """Debug the full conversion roundtrip for GPT-2."""
#         try:
#             hf_model, converted_model, hooked_model = self.debug_conversion_roundtrip()
            
#             # Test with multiple inputs to ensure robustness
#             test_texts = [
#                 "Hello world",
#                 "The quick brown fox",
#                 "Machine learning is",
#                 "Once upon a time"
#             ]
            
#             tokenizer = AutoTokenizer.from_pretrained("gpt2")
#             tokenizer.pad_token = tokenizer.eos_token
            
#             all_close = True
#             max_diff_overall = 0
            
#             for test_text in test_texts:
#                 tokens = tokenizer(test_text, return_tensors="pt")
                
#                 with torch.no_grad():
#                     hf_output = hf_model(**tokens).logits
#                     converted_output = converted_model(**tokens).logits
                
#                 max_diff = torch.max(torch.abs(hf_output - converted_output)).item()
#                 mean_diff = torch.mean(torch.abs(hf_output - converted_output)).item()
                
#                 print(f"'{test_text}': max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
                
#                 if max_diff > 1e-4:
#                     all_close = False
#                     max_diff_overall = max(max_diff_overall, max_diff)
            
#             # For now, use relaxed assertion to see the scale of differences
#             assert max_diff_overall < 1.0, f"Conversion differences too large: {max_diff_overall:.6f}"
            
#             print(f"✅ Roundtrip conversion completed with max difference: {max_diff_overall:.6f}")
            
#         except Exception as e:
#             pytest.fail(f"Roundtrip debug failed: {e}")
    
#     @pytest.mark.integration
#     @pytest.mark.slow  
#     def test_gpt2_small_conversion_matches_hf(self):
#         """Test that GPT-2 small converted weights match original HF weights."""
#         model_name = "gpt2"
        
#         try:
#             print("Loading original HuggingFace model...")
#             hf_original = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
#             hf_original_state = hf_original.state_dict()
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
            
#             print("Loading HookedTransformer and converting weights...")
#             hooked_model = HookedTransformer.from_pretrained_no_processing(model_name)
#             hooked_state = hooked_model.state_dict()
            
#             # Check if unembed bias exists and handle it appropriately
#             if "unembed.b_U" in hooked_state:
#                 print("HookedTransformer has unembed.b_U, but HF doesn't use it")
#                 unembed_bias = hooked_state["unembed.b_U"]
#                 if torch.allclose(unembed_bias, torch.zeros_like(unembed_bias), atol=1e-6):
#                     print("✓ unembed.b_U is effectively zero (can be safely ignored)")
#                 else:
#                     print("⚠ unembed.b_U has non-zero values! This might cause issues")
            
#             # Convert to HF format
#             converted_state, missing = convert_hooked_to_gpt2_weights(hooked_state, hooked_model.cfg)
            
#             # For GPT-2, we expect unembed.b_U to be missing from the conversion
#             if "unembed.b_U" in missing:
#                 print("✓ unembed.b_U correctly excluded from HF conversion")
#                 missing.remove("unembed.b_U")  # Remove it from missing params since this is expected
            
#             assert len(missing) == 0, f"Unexpected missing params: {missing}"
            
#             print("Creating HF model with converted weights...")
#             hf_config = AutoConfig.from_pretrained(model_name)
#             hf_converted = AutoModelForCausalLM.from_config(hf_config).float()
            
#             # Use strict=False to handle any minor parameter differences
#             load_result = hf_converted.load_state_dict(converted_state, strict=False)
            
#             print(f"Load result - Missing keys: {load_result.missing_keys}")
#             print(f"Load result - Unexpected keys: {load_result.unexpected_keys}")
            
#             hf_converted_state = hf_converted.state_dict()
            
#             print("Testing forward pass consistency...")
#             test_text = "Hello, how are you today?"
#             inputs = tokenizer(test_text, return_tensors="pt")
            
#             with torch.no_grad():
#                 original_output = hf_original(**inputs)
#                 converted_output = hf_converted(**inputs)
            
#             # Convert both to same dtype for comparison
#             original_logits = original_output.logits.float()
#             converted_logits = converted_output.logits.float()
            
#             # Use more relaxed tolerance for initial testing
#             max_diff = torch.max(torch.abs(original_logits - converted_logits)).item()
#             mean_diff = torch.mean(torch.abs(original_logits - converted_logits)).item()
            
#             print(f"Max difference: {max_diff:.6f}")
#             print(f"Mean difference: {mean_diff:.6f}")
            
#             # Start with relaxed tolerance and tighten as you fix issues
#             assert torch.allclose(original_logits, converted_logits, atol=1e-2, rtol=1e-1), \
#                 f"Logits differ significantly (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})"
            
#             print("✅ GPT-2 conversion matches original HF model!")
            
#         except Exception as e:
#             print(f"GPT-2 test failed with error: {e}")
#             import traceback
#             traceback.print_exc()
#             pytest.fail(f"GPT-2 test failed: {e}")


# # Add this to your existing test file
# def test_debug_conversion():
#     """Run the debug conversion separately."""
#     test_class = TestRealModelConversion()
#     test_class.debug_conversion_roundtrip()


# if __name__ == "__main__":
#     # Run the debug function directly
#     test_debug_conversion()
