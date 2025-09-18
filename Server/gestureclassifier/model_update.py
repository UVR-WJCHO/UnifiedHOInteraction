import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder
import math


class Attention(nn.Module):
    def __init__(self, attention_dim, gru_dim=128):
        super(Attention, self).__init__()
        self.w = nn.Linear(attention_dim, attention_dim, bias=False)
        self.gru = nn.GRU(gru_dim, gru_dim, 1, batch_first=True)

    def forward(self, input_padded, hidden):
        e = torch.bmm(self.w(input_padded), hidden.permute(1, 2, 0))
        context = torch.bmm(input_padded.permute(0, 2, 1), e.softmax(dim=1))
        context = context.permute(0, 2, 1)

        # Compute the auxiliary context, and concat
        aux_context, _ = self.gru(context, hidden)
        output = torch.cat([aux_context, context], 2).squeeze(1)

        return output

class SelfAttention(nn.Module):
    """Self-Attention mechanism for sequence modeling"""

    def __init__(self, hidden_dim, dropout=0.2):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Linear layers for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Output projection
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        # batch_size, seq_len, hidden_dim = x.shape

        # Generate Q, K, V
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)  # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, seq_len, seq_len]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, seq_len, hidden_dim]

        # Output projection with residual connection
        output = self.output(context) + x
        output = self.norm(output)

        return output

# ----------------------------------------------------------------------------------------------------------------------
class DeepGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        self.gru1 = nn.GRU(self.num_features, 512, 2, batch_first=True)
        self.gru2 = nn.GRU(512, 256, 2, batch_first=True)
        self.gru3 = nn.GRU(256, 128, 1, batch_first=True)

        # Attention
        self.attention = Attention(128)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_padded):
        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, _ = self.gru1(x_packed)
        output, _ = self.gru2(output)
        output, hidden = self.gru3(output)

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ----------------------------------------------------------------------------------------------------------------------


class DeepGRU_R(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepGRU_R, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(128)
        self.gru_dropout = nn.Dropout(0.2)

        # Attention
        self.attention = Attention(128)
        # self.attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_padded):
        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        x_packed = packer(x_padded, x_lengths.cpu(), batch_first=True)

        # Encode
        output, hidden = self.gru1(x_packed)
        # output = self.gru_dropout(self.gru_norm(output))

        # Pass to attention with the original padding
        output_padded, _ = padder(output, batch_first=True)
        attn_output = self.attention(output_padded, hidden[-1:])

        # Classify
        return self.classifier(attn_output)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DeepGRU_RS(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepGRU_RS, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 256, 2, batch_first=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Attention
        # self.attention = Attention(128)
        self.attention = SelfAttention(hidden_dim=256, dropout=0.1)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_padded):
        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        # Encode
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        output = self.attention(output)
        output_transposed = output.transpose(1, 2)
        avg_pooled = self.adaptive_pool(output_transposed).squeeze(-1)

        # Classify
        return self.classifier(avg_pooled)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DeepGRU_RSB(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepGRU_RSB, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Attention
        # self.attention = Attention(128)
        self.attention = SelfAttention(hidden_dim=256, dropout=0.1)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_padded):
        x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        # Encode
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        output = self.attention(output)
        output_transposed = output.transpose(1, 2)
        avg_pooled = self.adaptive_pool(output_transposed).squeeze(-1)

        # Classify
        return self.classifier(avg_pooled)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedDeepGRU(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EnhancedDeepGRU, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Attention
        # self.attention = Attention(128)
        self.attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )

        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  # 최종 15클래스
        )

    def forward(self, x_padded):
        # x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        # Encode
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        output = self.attention(output)
        output_transposed = output.transpose(1, 2)

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(output_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(output_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(output)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(output * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnhancedDeepGRU_enh(nn.Module):
    def __init__(self, num_features, num_classes):
        super(EnhancedDeepGRU_enh, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Encoder
        # Bidirectional GRU for better context (single layer for speed)
        self.gru1 = nn.GRU(num_features, 128, 2, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_norm = nn.LayerNorm(256)
        self.gru_dropout = nn.Dropout(0.2)

        # Attention
        # self.attention = Attention(128)
        self.attention = SelfAttention(hidden_dim=256, dropout=0.1)

        # Feature enhancement (lightweight)
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 192, bias=False),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 256, bias=False)
        )
        # Efficient pooling for fixed sequence length (10)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Learnable attention pooling (more efficient for fixed length)
        self.attention_pool = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False)
        )

        # # 수정: 15클래스에 최적화된 3층 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Dropout(0.35),
            nn.Linear(256 * 3, 384),  # 더 넓은 첫 번째 레이어
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),  # 점진적 감소
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 64),  # 15클래스에 맞는 적절한 압축
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)  # 최종 15클래스
        )

    def forward(self, x_padded):
        # x_lengths = torch.full((x_padded.shape[0],), x_padded.shape[1]).cpu()

        # Encode
        output, _ = self.gru1(x_padded)
        output = self.gru_dropout(self.gru_norm(output))

        output = self.attention(output)

        #  Feature enhancement with skip connection
        enhanced = self.feature_enhance(output)
        enhanced = enhanced + output  # Skip connection

        output_transposed = enhanced.transpose(1, 2)

        # 1. Average pooling
        avg_pooled = self.adaptive_pool(output_transposed).squeeze(-1)  # [batch_size, 256]
        # 2. Max pooling
        max_pooled = self.max_pool(output_transposed).squeeze(-1)  # [batch_size, 256]
        # 3. Learnable attention pooling (no masking needed)
        attention_scores = self.attention_pool(output)  # [batch_size, 10, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, 10, 1]
        attention_pooled = torch.sum(output * attention_weights, dim=1)  # [batch_size, 256]

        # Combine all pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch_size, 768]

        # Classification
        return self.classifier(pooled_features)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------------------------------
def create_model(num_features=60, num_classes=15, model_opt=1):
    """
    Backward compatibility - creates fast model by default
    """
    if model_opt == 0:
        model = DeepGRU(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == 1:
        model = EnhancedDeepGRU(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == 2:
        model = DeepGRU_R(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == 3:
        model = DeepGRU_RS(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == 4:
        model = DeepGRU_RSB(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == 5:    # full kpt
        model = EnhancedDeepGRU(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == 6:    # partial kpt
        model = EnhancedDeepGRU(
            num_features=num_features,
            num_classes=num_classes,
        )
    elif model_opt == -1:
        model = EnhancedDeepGRU_enh(
            num_features=num_features,
            num_classes=num_classes,
        )


    print(f"Fast model with self-attention created with {model.get_num_params():,} parameters")
    return model


