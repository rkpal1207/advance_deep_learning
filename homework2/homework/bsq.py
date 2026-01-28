import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        # Number of bits used to represent each embedding in the binary codebook
        self._codebook_bits = codebook_bits

        # Binary encoder that maps continuous embeddings to a fixed-length binary representation
        self.binary_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, codebook_bits)
        )

        # the network invert the quantization more accurately
        self.binary_decoder = torch.nn.Sequential(
            torch.nn.Linear(codebook_bits, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Linear(embedding_dim, embedding_dim),
            # extra nonlinearity
            torch.nn.GELU(),              
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.GELU(),              
            torch.nn.Linear(embedding_dim, embedding_dim)
        )

        self.residual = True
        #raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        # Project embeddings into the binary codebook space
        x = self.binary_encoder(x)      
        # Normalize the projected vectors to unit length for stability                      
        x = torch.nn.functional.normalize(x, dim=-1)   
        # Apply differentiable sign function to obtain binary-like codes
        x = diff_sign(x)
        # Return the binary latent representation                               
        return x
        #raise NotImplementedError()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        decoded = self.binary_decoder(x)
        if self.residual:
            # residual connection: decoded + (x @ first layer's weights)
            return decoded + x @ self.binary_decoder[0].weight.T
        else:
            # output without residual connection
            return decoded
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)
        self.codebook_bits = codebook_bits
        #raise NotImplementedError()

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        continuous_latents = super().encode(x)         
        return self.bsq.encode_index(continuous_latents)
        #raise NotImplementedError()

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        continuous_latents = self.bsq.decode_index(x)    
        return super().decode(continuous_latents) 
        #raise NotImplementedError()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = super().encode(x)       
        return self.bsq.encode(features) 
        #raise NotImplementedError()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.bsq.decode(x)       
        return super().decode(decoded) 
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # Encode the input image into discrete tokens / latent representations
        tokens = self.encode(x)

        # Reconstruct the image from the encoded tokens
        recon_img = self.decode(tokens)

        # Encode the input into codebook indices for usage statistics
        indices = self.encode_index(x)

        # Count how often each codebook entry is used
        cnt = torch.bincount(
            indices.flatten(), 
            minlength=2**self.codebook_bits
            )

        # Compute codebook usage distribution (do not detach to allow gradients)
        p = cnt.float() / cnt.sum()

        # Entropy of codebook usage to encourage uniform utilization
        entropy_ls = -torch.sum(p * torch.log(p + 1e-8))

        return recon_img, {
            # Fraction of codebook entries that are never used
            "cb0":     (cnt == 0).float().mean().detach(), 
            # Fraction of codebook entries used very rarely (≤ 2 times)  
            "cb2":     (cnt <= 2).float().mean().detach(),  
            # Entropy regularization term (weighted, gradients flow through)
            "entropy": 0.01 * entropy_ls                     
        }
        #raise NotImplementedError()
