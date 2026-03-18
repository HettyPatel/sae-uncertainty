import torch
from sae_lens import SAE


def load_sae(config: dict, layer_idx: int, device: str = "cuda"):
    """Load a pretrained SAE for the given model config and layer."""
    release = config["sae_release"]
    sae_id = config["sae_id_template"].format(layer=layer_idx)
    print(f"  Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


class ResidualStreamHook:
    """Captures the residual stream (last token) after specified transformer layers."""

    def __init__(self):
        self.hidden_states = {}
        self.handles = []

    def make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.hidden_states[layer_idx] = output[0][:, -1, :].detach()
            else:
                self.hidden_states[layer_idx] = output[:, -1, :].detach()
        return hook_fn

    def register(self, model, model_config, layer_indices):
        from src.model import get_layers
        layers = get_layers(model, model_config)
        for i in layer_indices:
            handle = layers[i].register_forward_hook(self.make_hook(i))
            self.handles.append(handle)

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hidden_states = {}


class SAESuppressionHook:
    """Forward hook that suppresses specified SAE features via encode-modify-decode."""

    def __init__(self, sae, feature_indices, scale=0.0):
        self.sae = sae
        self.feature_indices = (
            torch.tensor(feature_indices, dtype=torch.long)
            if len(feature_indices) > 0
            else torch.tensor([], dtype=torch.long)
        )
        self.scale = scale
        self.handle = None

    def hook_fn(self, module, input, output):
        if len(self.feature_indices) == 0:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        last_token = hidden[:, -1:, :].float()

        with torch.no_grad():
            feat_acts = self.sae.encode(last_token)
            feat_acts_modified = feat_acts.clone()
            feat_acts_modified[:, :, self.feature_indices] *= self.scale
            reconstructed = self.sae.decode(feat_acts_modified)
            original_reconstructed = self.sae.decode(feat_acts)
            delta = reconstructed - original_reconstructed
            modified_last = hidden[:, -1:, :] + delta.to(hidden.dtype)

        hidden_modified = hidden.clone()
        hidden_modified[:, -1:, :] = modified_last

        if rest is not None:
            return (hidden_modified,) + rest
        return hidden_modified

    def register(self, model, model_config, layer_idx):
        from src.model import get_layers
        layers = get_layers(model, model_config)
        self.handle = layers[layer_idx].register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None
