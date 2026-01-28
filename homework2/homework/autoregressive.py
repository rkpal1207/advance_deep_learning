import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()

        # Latent embedding dimension and vocabulary size
        self.d_latent = d_latent
        self.n_tokens = n_tokens

       # Token embedding layer: maps token indices to latent vectors
        self.token_emb = torch.nn.Embedding(
            num_embeddings=self.n_tokens,
            embedding_dim=self.d_latent
            )
        
        # Learnable positional embeddings (max sequence length = 1000)
        pos_tensor = torch.zeros(1, 1000, self.d_latent)
        self.pos_emb = torch.nn.Parameter(pos_tensor)

        n_heads = 8 
        n_layers = 6 

        # Transformer configuration
        encoder_config = torch.nn.TransformerEncoderLayer(
            d_model=self.d_latent,
            nhead=n_heads,
            dim_feedforward=self.d_latent * 4,  
            activation='gelu'                   
            )
        
        # Stacked transformer encoder
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=encoder_config,
            num_layers=n_layers
            )
        
        # Output projection: map latent states to token logits
        self.to_log = torch.nn.Linear(
            in_features=self.d_latent,
            out_features=self.n_tokens
            )
        #raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        batch_size, height, width = x.shape
        seq_len = height * width

        # Flatten spatial grid into a token sequence
        # (B, H, W) -> (B, seq_len)
        t_seq = x.view(batch_size, seq_len)

        # Convert token indices to embeddings
        # (B, seq_len) -> (B, seq_len, d_latent)
        t_embed = self.token_emb(t_seq)  

        # Scale embeddings by sqrt(d_latent) for training stability (Transformer convention) -> recommend by ChatGPT    
        scaled_embed = t_embed * torch.sqrt(torch.tensor(self.d_latent, dtype=torch.float32))
        # Add positional embeddings (truncate to sequence length)
        pos_embed = self.pos_emb[:, :seq_len, :] 
        combined_embed = scaled_embed + pos_embed
        
        # Create causal attention mask to prevent attending to future tokens
        # Upper triangular matrix with -inf above the diagonal
        mask = torch.ones(seq_len, seq_len) * float('-inf')
        mask = torch.triu(mask, diagonal=1).to(combined_embed.device)
        
        # Transformer expects input as (seq_len, batch_size, d_latent)
        # # (B, seq_len, d_latent) -> (seq_len, B, d_latent)
        transformer_input = combined_embed.permute(1, 0, 2)
        transformer_output = self.transformer(transformer_input, mask=mask)
        # Restore batch-first format
        # (seq_len, B, d_latent) -> (B, seq_len, d_latent)
        transformer_output = transformer_output.permute(1, 0, 2)

        t_log = self.to_log(transformer_output) 
        spatial_log = t_log.view(batch_size, height, width, self.n_tokens)
        # Return spatial logits (no auxiliary losses)
        return spatial_log, {}
        #raise NotImplementedError()

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        # Use provided device or fall back to the model's parameter device
        device = device or next(self.parameters()).device
        # Total number of tokens in the spatial grid
        # Initialize sequence with zeros (start tokens)
        # Shape: (B, seq_len)
        seq_len = h * w                           
        seq = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        # Autoregressive sampling loo
        for i in range(seq_len):
            logits, _ = self.forward(seq.view(B, h, w))
            flat = logits.view(B, seq_len, self.n_tokens)
            probs = torch.softmax(flat[:, i, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            seq[:, i] = next_token

        # Reshape generated sequence back to spatial grid
        # (B, seq_len) -> (B, h, w)
        return seq.view(B, h, w)
        #raise NotImplementedError()
