import math
import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (seq_len, seq_len)
    attn = scores.softmax(dim=-1)                       # (seq_len, seq_len)，每行和=1
    return attn @ V                                     # (seq_len, d_v)


if __name__ == "__main__":
    # 注意：用 float 而不是 int，否则 softmax 会报类型错误
    Q = torch.tensor([[1., 0.], [0., 1.]])
    K = torch.tensor([[1., 0.], [0., 1.]])
    V = torch.tensor([[1., 2.], [3., 4.]])

    out = self_attention(Q, K, V)
    print("output =")
    print(out)

    expected = torch.tensor([[1.660477, 2.660477],
                             [2.339523, 3.339523]])
    print("\nmatches expected:", torch.allclose(out, expected, atol=1e-4))