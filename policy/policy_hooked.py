import torch
from typing import Optional
from rl4co.models.zoo.am.policy import AttentionModelPolicy

class HookedAttentionModelPolicy(AttentionModelPolicy):
    """
    Extension of AttentionModelPolicy that allows registering hooks for mechanistic interpretability.
    Adds functionality to capture intermediate activations from the encoder layers.
    """
    def __init__(self, *args, **kwargs):
        # Pass all kwargs to the parent class (including any dropout parameters)
        super().__init__(*args, **kwargs)
        self.hooks = {}
        self.activation_cache = {}
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup the hooks for each encoder layer"""
        for layer_idx, layer in enumerate(self.encoder.net.layers):
            # Register hook for the output of each encoder layer
            def get_hook(layer_idx):
                def hook(module, input, output):
                    self.activation_cache[f'encoder_layer_{layer_idx}'] = output
                return hook
            
            # Register the hook on the MultiHeadAttentionLayer
            handle = layer.register_forward_hook(get_hook(layer_idx))
            self.hooks[f'encoder_layer_{layer_idx}'] = handle
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
    
    def clear_cache(self):
        """Clear the activation cache"""
        self.activation_cache.clear()
    
    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """
        Retrieve activation from cache by name
        
        Args:
            name: Name of the activation to retrieve (e.g., 'encoder_layer_0')
        
        Returns:
            torch.Tensor or None if activation not found
        """
        return self.activation_cache.get(name)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that automatically clears the activation cache before each run
        """
        self.clear_cache()
        return super().forward(*args, **kwargs)


class EnhancedHookedPolicy(HookedAttentionModelPolicy):
    """Hooked policy that also captures encoder output and decoder input."""

    def _setup_hooks(self):
        super()._setup_hooks()

        def encoder_output_hook(_module, _input, output):
            self.activation_cache["encoder_output"] = output

        def decoder_input_hook(_module, input, _output):
            if isinstance(input, tuple) and len(input) > 0:
                self.activation_cache["decoder_input"] = input[0]
            else:
                self.activation_cache["decoder_input"] = None

        self.hooks["encoder_output"] = self.encoder.register_forward_hook(encoder_output_hook)
        self.hooks["decoder_input"] = self.decoder.register_forward_hook(decoder_input_hook)
