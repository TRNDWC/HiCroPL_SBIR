# HiCroPL: Hierarchical Cross-modal Prompt Learning - Chi tiết Kiến trúc Trao đổi 2 Modality

## Tóm tắt

HiCroPL là một phương pháp học prompt cross-modal (Text-to-Image và Image-to-Text) trong CLIP. Mục tiêu là học các learnable token prompt cho cả text encoder và visual encoder sao cho chúng có thể trao đổi thông tin với nhau qua các tầng sâu của mô hình.

**Các thành phần chính:**
- **Text Modality**: Learnable text prompts được nhúng vào các tầng text encoder
- **Visual Modality**: Learnable visual prompts được nhúng vào các tầng visual encoder  
- **Cross-Modal Exchange**: Các module ánh xạ trao đổi thông tin giữa 2 modality (T→I, I→T)

---

## 1. Các Lớp Cơ Bản Hỗ Trợ Cross-Modal Communication

### 1.1 AttentionPooling - Nén Thông Tin Bằng Attention

**File**: [trainers/hicropl.py](trainers/hicropl.py#L131)

```python
class AttentionPooling(nn.Module):
    """
    Nén một chuỗi token bằng cách sử dụng Attention mechanism.
    
    Mục đích: Giảm chiều dữ liệu từ [n_ctx, hidden_size] xuống [1, hidden_size]
    bằng cách tập trung thông tin quan trọng vào một token duy nhất (query token).
    """
    def __init__(self, hidden_size, num_attention_heads):
        super(AttentionPooling, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        """
        Args:
            token_query: [1, hidden_size] - Token duy nhất sẽ nén thông tin
            sequence_key: [n_ctx, hidden_size] - Các token cần nén
            sequence_value: [n_ctx, hidden_size] - Các token cần nén
        
        Returns:
            token_query: [1, hidden_size] - Token đã được nén
        """
        # Attention mechanism để nén thông tin từ sequence vào token_query
        token_query = token_query + self.attn(
            self.ln_1(token_query),      # Query
            self.ln_1(sequence_key),     # Key
            self.ln_1(sequence_value),   # Value
            need_weights=False
        )[0]
        token_query = self.ln_2(token_query)
        return token_query
```

**Cách hoạt động:**
- Sử dụng Multihead Attention để tập trung n_ctx token thành 1 token
- Query được khởi tạo từ proxy token (text_proxy_token hoặc visual_proxy_token)
- Key và Value là các learnable prompt tokens
- Output là một token nén chứa thông tin tóm tắt của toàn bộ chuỗi

**Ví dụ sử dụng (trong CrossModalPromptLearner.forward()):**
```python
# Nén text prompts thành 1 token để truyền sang visual
for i in range(self.cross_layer):
    text_proxy_token = self.attn_pooling_text_nets[i](
        token_query=self.text_proxy_token[i],        # [1, 512]
        sequence_key=self.cross_prompts_text[i],      # [n_ctx, 512]
        sequence_value=self.cross_prompts_text[i]     # [n_ctx, 512]
    )  # Output: [1, 512]
```

---

### 1.2 CrossPromptAttention - Ánh xạ Thông Tin Giữa 2 Modality

**File**: [trainers/hicropl.py](trainers/hicropl.py#L144)

```python
class CrossPromptAttention(nn.Module):
    """
    Ánh xạ thông tin từ một modality sang modality khác.
    
    Mục đích: Có thể chuyển đổi thông tin Text→Visual hoặc Visual→Text
    bằng cách học một ánh xạ qua attention + feedforward network.
    """
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super(CrossPromptAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        
        # Các linear layer để ánh xạ Q, K, V về cùng chiều
        self.linear_q = nn.Linear(hidden_size, hidden_size)           # Query projection
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)   # Key projection
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)   # Value projection
        
        # Layer norm và feedforward
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(hidden_size * 4, hidden_size))
        ]))
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        """
        Cross-modal attention: Query từ modality A, Key/Value từ modality B
        
        Args:
            q: Query, shape [*, hidden_size] - Từ modality 1
            k: Key, shape [*, encoder_hidden_size] - Từ modality 2 
            v: Value, shape [*, encoder_hidden_size] - Từ modality 2
        
        Returns:
            Updated query: [*, hidden_size]
        """
        # Ánh xạ về cùng chiều
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        
        # Cross-modal attention: Q từ modality A, K,V từ modality B
        q_proj = q_proj + self.attn(
            self.ln_1(q_proj),
            self.ln_1(k_proj),
            self.ln_1(v_proj),
            need_weights=False
        )[0]
        
        # Feedforward network
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj
```

**Cách hoạt động:**
- **Text→Visual** (T→I): Query = visual prompts, Key/Value = text prompts
  - Visual prompts được cập nhật dựa trên thông tin từ text prompts
- **Visual→Text** (I→T): Query = text prompts, Key/Value = visual prompts
  - Text prompts được cập nhật dựa trên thông tin từ visual prompts

**Ví dụ sử dụng (T→I mapping):**
```python
# Từ CrossModalPromptLearner.forward()
visual_prompts = torch.cat([self.cross_prompts_visual[i].unsqueeze(0) 
                            for i in range(self.cross_layer)], dim=0)
# [self.cross_layer, n_ctx, 768]

proxy_text_prompts = torch.cat(proxy_text_tokens, dim=0)  # [self.cross_layer, 1, 512]

# Ánh xạ từ nén text prompts sang visual prompts
updated_visual_prompts = self.text2visual_net(
    visual_prompts,          # Q: visual prompts cần được cập nhật
    proxy_text_prompts,      # K: text prompts (nén)
    proxy_text_prompts       # V: text prompts (nén)
)  # Output: [self.cross_layer * n_ctx, 768]
```

---

### 1.3 CrossModalPromptLearner - Học Và Trao Đổi Prompt Giữa 2 Modality

**File**: [trainers/hicropl.py](trainers/hicropl.py#L169)

Đây là **lớp chính** điều phối toàn bộ quá trình học prompt cross-modal.

#### 1.3.1 Khởi tạo Learnable Prompts

```python
class CrossModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx = cfg.TRAINER.HICROPL.N_CTX                      # Số token prompt (ví dụ: 4)
        self.cross_prompts_depth = cfg.TRAINER.HICROPL.PROMPT_DEPTH  # Độ sâu (ví dụ: 3)
        self.cross_layer = cfg.TRAINER.HICROPL.CROSS_LAYER     # Số tầng T→I (ví dụ: 1)

        ######## Khởi tạo Text Prompts ########
        # Layer 0: Tầng đầu tiên của text encoder, shape [n_ctx, 512]
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Layer 1...n: Các tầng sâu hơn, shape [n_ctx, 512] cho mỗi tầng
        cross_prompts_text = nn.ParameterList([
            self.ctx,  # Tầng 0
            nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype)),  # Tầng 1
            nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype)),  # Tầng 2
            # ...
        ])
        self.cross_prompts_text = cross_prompts_text

        ######## Khởi tạo Visual Prompts ########
        # Mỗi tầng có n_ctx tokens, mỗi token có chiều 768 (visual dimension)
        cross_prompts_visual = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, 768, dtype=dtype)),  # Tầng 0
            nn.Parameter(torch.empty(n_ctx, 768, dtype=dtype)),  # Tầng 1
            nn.Parameter(torch.empty(n_ctx, 768, dtype=dtype)),  # Tầng 2
            # ...
        ])
        self.cross_prompts_visual = cross_prompts_visual

        ######## Cross-Modal Networks ########
        # Network để ánh xạ T→I
        self.text2visual_net = CrossPromptAttention(
            hidden_size=768,          # Q dimension (visual)
            encoder_hidden_size=512,  # K,V dimension (text)
            num_attention_heads=8
        )
        
        # Network để ánh xạ I→T
        self.visual2text_net = CrossPromptAttention(
            hidden_size=512,          # Q dimension (text)
            encoder_hidden_size=768,  # K,V dimension (visual)
            num_attention_heads=8
        )

        ######## Attention Pooling Networks (LKP) ########
        # Nén text prompts thành 1 token trước khi truyền sang visual
        attn_pooling_text = AttentionPooling(hidden_size=512, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_pooling_text, self.cross_layer)
        
        # Nén visual prompts thành 1 token trước khi truyền sang text
        attn_pooling_visual = AttentionPooling(hidden_size=768, num_attention_heads=8)
        self.attn_pooling_visual_nets = _get_clones(attn_pooling_visual, 
                                                    self.cross_prompts_depth - self.cross_layer)

        ######## Proxy Tokens ########
        # Các token này sẽ được sử dụng làm Query trong AttentionPooling
        text_proxy_token = torch.randn(1, 512, dtype=dtype)
        self.text_proxy_token = nn.ParameterList([
            nn.Parameter(text_proxy_token) for _ in range(self.cross_layer)
        ])
        
        visual_proxy_token = torch.randn(1, 768, dtype=dtype)
        self.visual_proxy_token = nn.ParameterList([
            nn.Parameter(visual_proxy_token) for _ in range(self.cross_layer, self.cross_prompts_depth)
        ])
```

**Cấu trúc Learnable Tokens:**

```
Giả sử: n_ctx=4, prompt_depth=3, cross_layer=1

Text Prompts (cross_prompts_text):
├── Layer 0: [4, 512] - Được truyền sang visual
├── Layer 1: [4, 512] - Nhận thông tin từ visual
└── Layer 2: [4, 512] - Nhận thông tin từ visual

Visual Prompts (cross_prompts_visual):
├── Layer 0: [4, 768] - Được cập nhật bằng T→I (từ layer 0 text)
├── Layer 1: [4, 768] - Được cập nhật bằng I→T (từ layer 0 visual)
└── Layer 2: [4, 768] - Được cập nhật bằng I→T (từ layer 0 visual)

Cross-Layer Boundary:
- Layers [0: self.cross_layer] (= 1): T→I mapping (Text tác động Visual)
- Layers [self.cross_layer: end] (= 1,2): I→T mapping (Visual tác động Text)
```

---

#### 1.3.2 Forward: Trao Đổi Prompt Giữa 2 Modality

**File**: [trainers/hicropl.py](trainers/hicropl.py#L305)

```python
def forward(self):
    """
    Luồng chính:
    1. Xây dựng text input từ learnable text prompts
    2. Text→Image (T→I) Mapping: Cập nhật visual prompts dựa trên text
    3. Image→Text (I→T) Mapping: Cập nhật text prompts dựa trên visual
    4. Trả về các prompts để được sử dụng trong encoder
    """
```

**STEP 1: Xây dựng Text Input Cho Tầng 0 Text Encoder**

```python
# Lấy text prompt của tầng 0
ctx = self.cross_prompts_text[0]  # [n_ctx, 512]
if ctx.dim() == 2:
    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, 512]

prefix = self.token_prefix      # [n_cls, 1, 512] - SOS token
suffix = self.token_suffix      # [n_cls, *, 512] - CLS, EOS tokens

# Xây dựng: [SOS, learnable_ctx, ..., CLS, EOS]
text_input = self.construct_prompts(ctx, prefix, suffix)  # [n_cls, 77, 512]
```

**STEP 2: Text→Image (T→I) Mapping**
Cập nhật visual prompts của các tầng [0, cross_layer) dựa trên text prompts

```python
# ===== T→I MAPPING =====
# Chỉ áp dụng cho self.cross_layer tầng đầu tiên (mặc định = 1)

# Lấy visual prompts của các tầng cần cập nhật
visual_prompts = torch.cat([
    self.cross_prompts_visual[i].unsqueeze(0) 
    for i in range(self.cross_layer)
], dim=0)  # [self.cross_layer, n_ctx, 768]

# Lấy text prompts tương ứng
text_prompts = torch.cat([
    self.cross_prompts_text[i].unsqueeze(0) 
    for i in range(self.cross_layer)
], dim=0)  # [self.cross_layer, n_ctx, 512]

# === LKP (Learnable Knowledge Pooling): Nén text prompts ===
proxy_text_tokens = []
for i in range(self.cross_layer):
    # Nén text_prompts[i] từ [n_ctx, 512] thành [1, 512]
    text_proxy_token = self.attn_pooling_text_nets[i](
        token_query=self.text_proxy_token[i],      # [1, 512] - Query token
        sequence_key=self.cross_prompts_text[i],    # [n_ctx, 512] - Tokens cần nén
        sequence_value=self.cross_prompts_text[i]   # [n_ctx, 512]
    )
    proxy_text_tokens.append(text_proxy_token)  # [1, 512]

proxy_text_prompts = torch.cat(proxy_text_tokens, dim=0)  
# [self.cross_layer, 1, 512]

# === Cross-Modal Attention: Ánh xạ T→I ===
# Reshape để text2visual_net xử lý
visual_prompts_flat = visual_prompts.view(-1, visual_prompts.shape[-1])
# [self.cross_layer * n_ctx, 768]

proxy_text_prompts_flat = proxy_text_prompts.view(-1, proxy_text_prompts.shape[-1])
# [self.cross_layer, 512]

# Cập nhật visual prompts dựa trên (nén) text prompts
updated_visual_prompts = self.text2visual_net(
    q=visual_prompts_flat,        # Query: visual prompts
    k=proxy_text_prompts_flat,    # Key: nén text prompts
    v=proxy_text_prompts_flat     # Value: nén text prompts
)  # [self.cross_layer * n_ctx, 768]

# Reshape lại và cập nhật
updated_visual_prompts = updated_visual_prompts.view(
    self.cross_layer, -1, updated_visual_prompts.shape[-1]
)  # [self.cross_layer, n_ctx, 768]

for i in range(self.cross_layer):
    self.cross_prompts_visual[i].data.copy_(updated_visual_prompts[i])
```

**STEP 3: Image→Text (I→T) Mapping**
Cập nhật text prompts của các tầng [cross_layer, end) dựa trên visual prompts

```python
# ===== I→T MAPPING =====
# Chỉ áp dụng cho các tầng [self.cross_layer: self.cross_prompts_depth]

# Lấy các prompts cần cập nhật
text_prompts = torch.cat([
    self.cross_prompts_text[i].unsqueeze(0) 
    for i in range(self.cross_layer, self.cross_prompts_depth)
], dim=0)  # [depth - cross_layer, n_ctx, 512]

visual_prompts = torch.cat([
    self.cross_prompts_visual[i].unsqueeze(0) 
    for i in range(self.cross_layer, self.cross_prompts_depth)
], dim=0)  # [depth - cross_layer, n_ctx, 768]

# === LKP: Nén visual prompts ===
proxy_visual_tokens = []
for i in range(self.cross_layer, self.cross_prompts_depth):
    visual_proxy_token = self.attn_pooling_visual_nets[i - self.cross_layer](
        token_query=self.visual_proxy_token[i - self.cross_layer],  # [1, 768]
        sequence_key=self.cross_prompts_visual[i],                   # [n_ctx, 768]
        sequence_value=self.cross_prompts_visual[i]                  # [n_ctx, 768]
    )
    proxy_visual_tokens.append(visual_proxy_token)  # [1, 768]

proxy_visual_prompts = torch.cat(proxy_visual_tokens, dim=0)
# [depth - cross_layer, 1, 768]

# === Cross-Modal Attention: Ánh xạ I→T ===
text_prompts_flat = text_prompts.view(-1, text_prompts.shape[-1])
# [(depth - cross_layer) * n_ctx, 512]

proxy_visual_prompts_flat = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])
# [(depth - cross_layer), 768]

# Cập nhật text prompts dựa trên (nén) visual prompts
updated_text_prompts = self.visual2text_net(
    q=text_prompts_flat,          # Query: text prompts
    k=proxy_visual_prompts_flat,  # Key: nén visual prompts
    v=proxy_visual_prompts_flat   # Value: nén visual prompts
)  # [(depth - cross_layer) * n_ctx, 512]

# Reshape lại và cập nhật
updated_text_prompts = updated_text_prompts.view(
    self.cross_prompts_depth - self.cross_layer, 
    -1, 
    updated_text_prompts.shape[-1]
)  # [depth - cross_layer, n_ctx, 512]

for i in range(self.cross_layer, self.cross_prompts_depth):
    self.cross_prompts_text[i].data.copy_(updated_text_prompts[i - self.cross_layer])
```

**STEP 4: Trả Về Prompts Cho Encoder**

```python
# Tách deeper prompts (bỏ tầng 0, vì tầng 0 là tầng đầu tiên)
cross_prompts_text_deeper = [
    self.cross_prompts_text[i] 
    for i in range(1, len(self.cross_prompts_text))
]  # [prompt_depth - 1] × [n_ctx, 512]

cross_prompts_visual_deeper = [
    self.cross_prompts_visual[i] 
    for i in range(1, len(self.cross_prompts_visual))
]  # [prompt_depth - 1] × [n_ctx, 768]

return (
    text_input,                          # [n_cls, 77, 512]
    self.cross_prompts_visual[0],        # [n_ctx, 768] 
    cross_prompts_text_deeper,           # List[Tensor]
    cross_prompts_visual_deeper          # List[Tensor]
)
```

---

## 2. Luồng Gọi Trong Training: Từ Prompt Learner Đến Encoder

### 2.1 CustomCLIP Forward Pass

**File**: [trainers/hicropl.py](trainers/hicropl.py#L412)

```python
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CrossModalPromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual  # VisionTransformer_HiCroPL
        self.text_encoder = TextEncoder(clip_model)  # Custom text encoder
        # ...

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # ===== STEP 1: Gọi Prompt Learner để trao đổi prompt =====
        text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper = \
            self.prompt_learner()
        # text_input: [n_cls, 77, 512] - text embeddings với learnable prompt
        # visual_ctx: [n_ctx, 768] - visual prompts cho tầng 0
        # cross_prompts_text_deeper: List[4] tensors [n_ctx, 512] - prompts cho tầng 1,2,3...
        # cross_prompts_visual_deeper: List[4] tensors [n_ctx, 768] - prompts cho tầng 1,2,3...

        # ===== STEP 2: Text Encoder với cross_prompts_text_deeper =====
        text_features = self.text_encoder(
            text_input,                    # [n_cls, 77, 512]
            tokenized_prompts,             # [n_cls, 77]
            cross_prompts_text_deeper      # List[...] - prompts cho các tầng sâu
        )  # Output: [n_cls, 512]

        # ===== STEP 3: Image Encoder với visual_ctx + cross_prompts_visual_deeper =====
        image_features = self.image_encoder(
            image.type(self.dtype),        # [batch_size, 3, 224, 224]
            visual_ctx,                    # [n_ctx, 768]
            cross_prompts_visual_deeper    # List[...] - prompts cho các tầng sâu
        )  # Output: [batch_size, 512]

        # ... normalize và tính logits ...
```

---

### 2.2 TextEncoder - Sử dụng Prompts Ở Tất Cả Các Tầng

**File**: [trainers/hicropl.py](trainers/hicropl.py#L107)

```python
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer  # Transformer encoder với các tầng
        # ...

    def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper):
        """
        Args:
            prompts: [n_cls, 77, 512] - Text embeddings + learnable prompts (tầu 0)
            tokenized_prompts: [n_cls, 77] - Chỉ số của các tokens
            cross_prompts_text_deeper: List[Tensor] - Prompts cho tầng 1,2,3...
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        # [n_cls, 77, 512]

        x = x.permute(1, 0, 2)  # NLD → LND = [77, n_cls, 512]

        # === BƯỚC QUAN TRỌNG: Gửi cả 2 thứ cho transformer ===
        combined = [x, cross_prompts_text_deeper]  # Đây là cải tiến HiCroPL!

        # Transformer sẽ xử lý x ở tất cả các tầng
        # và sử dụng cross_prompts_text_deeper để cập nhật prompts ở các tầng sâu
        outputs = self.transformer(combined)  
        # Output: [x, cross_prompts_text_deeper] (sau khi cập nhật)

        x = outputs[0]  # [77, n_cls, 512]
        x = x.permute(1, 0, 2)  # LND → NLD = [n_cls, 77, 512]
        x = self.ln_final(x).type(self.dtype)

        # Lấy feature từ end-of-text token (EOS)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # Output: [n_cls, 512] - Cuối cùng là text features

        return x
```

---

### 2.3 VisionTransformer_HiCroPL - Sử dụng Visual Prompts Ở Tất Cả Các Tầng

**File**: [clip/model.py](clip/model.py#L505)

```python
class VisionTransformer_HiCroPL(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, 
                 layers: int, heads: int, output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        # ... các parameter khác ...
        self.transformer = Transformer(
            width, layers, heads,
            prompts_needed=design_details["vision_depth"],  # Số tầng với prompts
            design_details=design_details
        )

    def forward(self, x: torch.Tensor, img_prompts, cross_prompts_visual_deeper):
        """
        Args:
            x: [batch_size, 3, 224, 224] - Hình ảnh đầu vào
            img_prompts: [n_ctx, 768] - Visual prompts cho tầng 0
            cross_prompts_visual_deeper: List[Tensor] - Visual prompts cho tầng 1,2,...
        
        Returns:
            [batch_size, 512] - Visual features
        """
        # === Patch Embedding ===
        x = self.conv1(x)  # [batch_size, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch_size, 768, 196]
        x = x.permute(0, 2, 1)  # [batch_size, 196, 768]

        # === Thêm Class Token ===
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)  # [batch_size, 197, 768] (1 CLS + 196 patches)

        # === Thêm Positional Embeddings ===
        x = x + self.positional_embedding.to(x.dtype)  # [batch_size, 197, 768]

        # === Thêm Visual Prompts Của Tầng 0 ===
        visual_ctx = img_prompts.expand(x.shape[0], -1, -1).half()
        # [batch_size, n_ctx, 768]

        x = torch.cat([x, visual_ctx], dim=1)
        # [batch_size, 197 + n_ctx, 768]

        # === Layer Norm & Permute ===
        x = self.ln_pre(x)  # Layer norm
        x = x.permute(1, 0, 2)  # NLD → LND = [197 + n_ctx, batch_size, 768]

        # === BƯỚC QUAN TRỌNG: Gửi cả 2 thứ cho transformer ===
        outputs = self.transformer([x, cross_prompts_visual_deeper])
        # [x, cross_prompts_visual_deeper] được cập nhật ở các tầng

        x = outputs[0]  # [197 + n_ctx, batch_size, 768]
        x = x.permute(1, 0, 2)  # LND → NLD = [batch_size, 197 + n_ctx, 768]

        # === Lấy CLS Token (CLS là token đầu tiên) ===
        x = self.ln_post(x[:, 0, :])  # [batch_size, 768]

        # === Project ===
        if self.proj is not None:
            x = x @ self.proj  # [batch_size, 512]

        return x
```

---

## 3. Các Tầng Transformer Xử Lý Prompts

### 3.1 Transformer Class - Lựa Chọn Block Theo Trainer

**File**: [clip/model.py](clip/model.py#L390)

```python
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, 
                 attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers

        # === Lựa chọn loại block dựa trên trainer ===
        current_trainer = design_details['trainer']  # 'HiCroPL'

        if current_trainer == 'HiCroPL' or current_trainer == 'MPT':
            # Tạo nn.Sequential với các ResidualAttentionBlock_HiCroPL
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock_HiCroPL(
                    width, heads, attn_mask,
                    add_prompt=(prompts_needed > i),  # Có thêm prompts hay không?
                    text_layer=text_layer,
                    i=i,
                    design_details=design_details
                )
                for i in range(layers)
            ])

    def forward(self, x: torch.Tensor):
        """
        x có thể là:
        - Tensor thông thường: [seq_len, batch, hidden] (cho IVLP)
        - List [x, cross_prompts_deeper]: x + learnable prompts (cho HiCroPL)
        """
        return self.resblocks(x)
```

**Ý nghĩa của `prompts_needed`:**
```
prompts_needed = design_details["vision_depth"] hoặc ["language_depth"]
                = cfg.TRAINER.HICROPL.PROMPT_DEPTH  (ví dụ: 3)

Nếu prompts_needed = 3 (độ sâu 3):
- Tầng 0 (i=0): prompts_needed > 0 → add_prompt=True
- Tầng 1 (i=1): prompts_needed > 1 → add_prompt=True
- Tầng 2 (i=2): prompts_needed > 2 → add_prompt=True
- Tầng 3+ (i>=3): prompts_needed > i → add_prompt=False

Điều này có nghĩa: Chỉ các tầng 0,1,2 sẽ thêm/cập nhật learnable prompts.
```

---

### 3.2 ResidualAttentionBlock_HiCroPL - Thêm & Cập Nhật Prompts Ở Mỗi Tầng

**File**: [clip/model.py](clip/model.py#L260)

```python
class ResidualAttentionBlock_HiCroPL(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 add_prompt=False, text_layer=False, i=0, design_details=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        # === Các tham số cho prompt handling ===
        self.text_layer = text_layer              # Là text layer hay visual layer?
        self.attn_mask = attn_mask
        self.cross_prompt_nctx = design_details['vision_ctx']  # Số prompts (ví dụ: 4)
        self.i = i                                 # Chỉ số tầng hiện tại
        
        # Chỉ thêm prompt nếu i != 0 (vì tầng 0 đã thêm ở input)
        if self.i != 0:
            self.add_prompt = add_prompt
        else:
            self.add_prompt = False

    def forward(self, inputs):
        """
        Nhận input từ ResidualAttentionBlock trước (hoặc từ transformer input).
        
        Input format:
        - inputs[0] (x): [seq_len, batch_size, hidden_dim]
          - Có thể chứa: [original_tokens, prompts_from_previous_layer]
        - inputs[1] (cross_prompts_deeper): List[Tensor] - Prompts cho các tầng sâu hơn
        """
        x = inputs[0]  # [seq_len, batch_size, hidden_dim]
        cross_prompts_deeper = inputs[1]  # List of Tensors

        # === STEP 1: Cập Nhật Prompts (Nếu cần) ===
        if self.add_prompt:
            if not self.text_layer:  # === VISUAL LAYER ===
                """
                Visual input structure:
                x = [patches_and_cls, prompts_from_previous_layer]
                    = [0:197, 197:197+n_ctx]
                
                Chúng ta sẽ:
                1. Lấy phần non-prompt: x[0:x.shape[0] - n_ctx]
                2. Lấy learnable prompts từ cross_prompts_deeper[i-1]
                3. Ghép lại
                """
                # Bỏ prompts của tầng trước
                prefix = x[0:x.shape[0] - self.cross_prompt_nctx, :, :]
                # [197, batch_size, 768] (bỏ n_ctx token cuối)

                # Lấy learnable prompts cho tầng này từ CrossModalPromptLearner
                visual_context = cross_prompts_deeper[self.i - 1]  # [n_ctx, 768]
                visual_context = visual_context.expand(x.shape[1], -1, -1)  # [n_ctx, batch, 768]
                visual_context = visual_context.permute(1, 0, 2)  # [batch, n_ctx, 768]
                visual_context = visual_context.half()  # Float16 optimization

                # Permute để phù hợp với seq_len-first format
                visual_context = visual_context.permute(1, 0, 2)  # [n_ctx, batch, 768]

                # Ghép visual prompts mới vào
                x = torch.cat([prefix, visual_context], dim=0)
                # [197 + n_ctx, batch_size, 768]

            else:  # === TEXT LAYER ===
                """
                Text input structure:
                x = [SOS_token, prompts_from_previous_layer, class_tokens, EOS_tokens]
                    = [0:1, 1:1+n_ctx, 1+n_ctx:77]
                
                Chúng ta sẽ:
                1. Lấy SOS token: x[:1]
                2. Lấy learnable prompts từ cross_prompts_deeper[i-1]
                3. Lấy suffix (class + EOS): x[1+n_ctx:]
                4. Ghép: [SOS, prompts, suffix]
                """
                prefix = x[:1, :, :]  # [1, batch_size, 512] - SOS token
                suffix = x[1 + self.cross_prompt_nctx:, :, :]  # [76, batch, 512] - class + EOS

                # Lấy learnable prompts cho tầng này
                textual_context = cross_prompts_deeper[self.i - 1]  # [n_ctx, 512]
                textual_context = textual_context.expand(x.shape[1], -1, -1)  # [n_ctx, batch, 512]
                textual_context = textual_context.permute(1, 0, 2)  # [batch, n_ctx, 512]
                textual_context = textual_context.half()

                textual_context = textual_context.permute(1, 0, 2)  # [n_ctx, batch, 512]

                # Ghép text prompts mới vào
                x = torch.cat([prefix, textual_context, suffix], dim=0)
                # [1 + n_ctx + 76, batch_size, 512] = [77, batch, 512]

        # === STEP 2: Self-Attention + MLP (Standard Transformer Block) ===
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        # === STEP 3: Trả Về Cho Tầng Tiếp Theo ===
        return [x, cross_prompts_deeper]
        # [x] chứa updated features (có prompts mới được nhúng vào)
        # [cross_prompts_deeper] được truyền tiếp (không thay đổi ở đây)
```

---

## 4. Sơ Đồ Luồng Dữ Liệu Chi Tiết

### 4.1 Luồng Text Encoding

```
STEP 1: prompt_learner()
═══════════════════════════════════════════════════════════════════
TextPrompts[0]       [n_ctx, 512]
TextPrompts[1:3]     List[n_ctx, 512]
           ↓
      Cross-Modal Exchange (T→I, I→T)
           ↓
       text_input        [n_cls, 77, 512] ← SOS + TextPrompts[0] + suffix
cross_prompts_text_deeper [3] × [n_ctx, 512] ← Updated TextPrompts[1:3]

STEP 2: text_encoder(text_input, tokenized_prompts, cross_prompts_text_deeper)
═══════════════════════════════════════════════════════════════════
text_input = [n_cls, 77, 512]
      ↓
Add Positional Embeddings
      ↓
x = [77, n_cls, 512]  (permute NLD→LND)
      ↓
Transformer([x, cross_prompts_text_deeper])

LAYER 0 (i=0, add_prompt=False):
────────────────────────────────
  Input: x = [77, n_cls, 512] từ positional embedding
  - Không thêm prompts (vì i=0)
  Output: [77, n_cls, 512]

LAYER 1 (i=1, add_prompt=True):
────────────────────────────────
  Input: x = [77, n_cls, 512]
  Thay prompts:
    - Remove: x[1+n_ctx:] = [76, n_cls, 512] (bỏ n_ctx prompts cũ)
    - Add: cross_prompts_text_deeper[0] → [n_ctx, n_cls, 512]
    - Result: x = [SOS(1) + new_prompts(n_ctx) + suffix(76), n_cls, 512]
  Output: [77, n_cls, 512]

LAYER 2 (i=2, add_prompt=True):
────────────────────────────────
  Input: x = [77, n_cls, 512]
  Thay prompts:
    - Remove: x[1+n_ctx:] = [76, n_cls, 512]
    - Add: cross_prompts_text_deeper[1] → [n_ctx, n_cls, 512]
    - Result: x = [SOS(1) + new_prompts(n_ctx) + suffix(76), n_cls, 512]
  Output: [77, n_cls, 512]

LAYER 3 (i=3, add_prompt=True):
────────────────────────────────
  Input: x = [77, n_cls, 512]
  Thay prompts:
    - Remove: x[1+n_ctx:] = [76, n_cls, 512]
    - Add: cross_prompts_text_deeper[2] → [n_ctx, n_cls, 512]
    - Result: x = [SOS(1) + new_prompts(n_ctx) + suffix(76), n_cls, 512]
  Output: [77, n_cls, 512]

LAYER 12 (i=12, add_prompt=False):
──────────────────────────────────
  Input: x = [77, n_cls, 512]
  - Không thêm/thay prompts
  Output: [77, n_cls, 512]
      ↓
Permute & LayerNorm
      ↓
x = [n_cls, 77, 512]
      ↓
Extract EOS token @ index[argmax(tokenized_prompts)]
      ↓
text_features = [n_cls, 512]
```

---

### 4.2 Luồng Visual Encoding

```
STEP 1: prompt_learner()
═══════════════════════════════════════════════════════════════════
VisualPrompts[0]       [n_ctx, 768]
VisualPrompts[1:3]     List[n_ctx, 768]  (updated by cross-modal exchange)
           ↓
       visual_ctx        [n_ctx, 768] ← VisualPrompts[0]
cross_prompts_visual_deeper [3] × [n_ctx, 768] ← VisualPrompts[1:3]

STEP 2: image_encoder(image, visual_ctx, cross_prompts_visual_deeper)
═══════════════════════════════════════════════════════════════════
image = [batch_size, 3, 224, 224]
      ↓
Patch Embedding (Conv2d)
      ↓
x = [batch_size, 768, 14, 14]
      ↓
Reshape: [batch_size, 196, 768]  (196 = 14×14 patches)
      ↓
Add CLS token: [batch_size, 197, 768]
      ↓
Add Positional Embeddings
      ↓
Add visual_ctx (Tầu 0 prompts)
      ↓
x = [batch_size, 197 + n_ctx, 768]  (e.g., [batch, 201, 768] if n_ctx=4)
      ↓
x = x.permute(1, 0, 2)  → [201, batch_size, 768]
      ↓
Transformer([x, cross_prompts_visual_deeper])

LAYER 0 (i=0, add_prompt=False):
────────────────────────────────
  Input: x = [201, batch_size, 768] (từ input + visual_ctx)
  - Không thêm prompts (vì visual prompts đã thêm ở input)
  Output: [201, batch_size, 768]

LAYER 1 (i=1, add_prompt=True):
────────────────────────────────
  Input: x = [201, batch_size, 768]
  Thay prompts:
    - Prefix (non-prompt): x[0:197] = [197, batch, 768]
    - New prompts: cross_prompts_visual_deeper[0] → [n_ctx, batch, 768]
    - Result: x = [197 + n_ctx, batch_size, 768]
  Output: [201, batch_size, 768]

LAYER 2 (i=2, add_prompt=True):
────────────────────────────────
  Input: x = [201, batch_size, 768]
  Thay prompts:
    - Prefix: x[0:197]
    - New prompts: cross_prompts_visual_deeper[1]
    - Result: x = [197 + n_ctx, batch_size, 768]
  Output: [201, batch_size, 768]

LAYER 3 (i=3, add_prompt=True):
────────────────────────────────
  Input: x = [201, batch_size, 768]
  Thay prompts:
    - Prefix: x[0:197]
    - New prompts: cross_prompts_visual_deeper[2]
    - Result: x = [197 + n_ctx, batch_size, 768]
  Output: [201, batch_size, 768]

LAYER 12 (i=12, add_prompt=False):
──────────────────────────────────
  Input: x = [201, batch_size, 768]
  - Không thêm/thay prompts
  Output: [201, batch_size, 768]
      ↓
Permute: [batch_size, 201, 768]
      ↓
LayerNorm & Extract CLS token (token đầu tiên, i=0)
      ↓
x = x[:, 0, :] = [batch_size, 768]
      ↓
Project: x @ self.proj → [batch_size, 512]
      ↓
image_features = [batch_size, 512]
```

---

## 5. Ví Dụ Cụ Thể: Forward Pass Hoàn Chỉnh

```python
# ===== FORWARD PASS EXAMPLE =====
n_cls = 10                 # 10 classes
batch_size = 32
n_ctx = 4                  # 4 learnable tokens per layer
prompt_depth = 3           # 3 layers with learnable prompts
cross_layer = 1            # First layer is T→I, rest is I→T

# ===== STEP 1: CrossModalPromptLearner =====
text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper = prompt_learner()

Outputs:
├── text_input: [10, 77, 512]         - n_cls × full_seq × embedding_dim
├── visual_ctx: [4, 768]              - n_ctx × visual_dim
├── cross_prompts_text_deeper:        - Danh sách 3 tensors
│   ├── [0]: [4, 512]  - Cho layer 1 của text encoder
│   ├── [1]: [4, 512]  - Cho layer 2 của text encoder
│   └── [2]: [4, 512]  - Cho layer 3 của text encoder
└── cross_prompts_visual_deeper:      - Danh sách 3 tensors
    ├── [0]: [4, 768]  - Cho layer 1 của visual encoder
    ├── [1]: [4, 768]  - Cho layer 2 của visual encoder
    └── [2]: [4, 768]  - Cho layer 3 của visual encoder

# ===== STEP 2: Text Encoder =====
text_features = text_encoder(text_input, tokenized_prompts, cross_prompts_text_deeper)
# [10, 512]

# ===== STEP 3: Image Encoder =====
image_features = image_encoder(image, visual_ctx, cross_prompts_visual_deeper)
# [32, 512]

# ===== STEP 4: Normalize & Compute Logits =====
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
# [32, 512] (normalized)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)
# [10, 512] (normalized)

logits = logit_scale * image_features @ text_features.t()
# [32, 10] - Mỗi hình ảnh so sánh với 10 classes
```

---

## 6. Tóm Tắt Kiến Trúc Cross-Modal

| Thành phần | Mục đích | Input | Output |
|-----------|---------|-------|--------|
| **AttentionPooling** | Nén chuỗi token thành 1 token | [n_ctx, dim] | [1, dim] |
| **CrossPromptAttention** | Ánh xạ thông tin giữa 2 modality | Q,K,V shape khác nhau | [*, Q_dim] |
| **CrossModalPromptLearner** | Điều phối trao đổi prompt | - | text/visual prompts |
| **ResidualAttentionBlock_HiCroPL** | Thêm/cập nhật prompts ở mỗi tầng | [x, cross_prompts] | [x, cross_prompts] |
| **TextEncoder** | Mã hóa text với prompts | text_input + cross_prompts | [n_cls, 512] |
| **VisionTransformer_HiCroPL** | Mã hóa hình ảnh với prompts | image + prompts | [batch, 512] |

---

## 7. Các Điểm Quan Trọng

### 7.1 Tại Sao Cần Cross-Modal Exchange?
- **T→I Mapping** (Text→Image): Text prompts mang thông tin về "nghĩa" của từ, có thể giúp visual prompts "hiểu" bối cảnh ngôn ngữ
- **I→T Mapping** (Image→Text): Visual prompts mang thông tin về "hình ảnh", có thể giúp text prompts "hiểu" bối cảnh trực quan

### 7.2 Learnable Knowledge Pooling (LKP)
- Trước khi truyền thông tin giữa 2 modality, thông tin được **nén** từ [n_ctx, dim] thành [1, dim]
- Việc nén này giúp:
  - Giảm overhead tính toán
  - Tập trung thông tin quan trọng
  - Tránh "đầu nhiễu" từ quá nhiều token

### 7.3 Layer Separation (cross_layer)
- Các tầng được chia làm 2 phần:
  - **T→I** (layers 0 to cross_layer-1): Text ảnh hưởng tới visual
  - **I→T** (layers cross_layer to end): Visual ảnh hưởng tới text
- Điều này cho phép mô hình học cách tương tác hai chiều giữa 2 modality

### 7.4 Why List Input Format?
```python
# ResidualAttentionBlock_HiCroPL.forward() nhận:
inputs = [x, cross_prompts_deeper]

# Thay vì:
outputs = self.resblocks(x)

# Lý do: nn.Sequential không thể truyền list qua các layer, nên
# mỗi layer nhận list [x, cross_prompts_deeper] và trả list [x, cross_prompts_deeper]
```

---

## 8. Kết Luận

**HiCroPL kiến trúc cross-modal hoạt động theo chu trình:**

1. **CrossModalPromptLearner** học prompts cho cả 2 modality
2. **Text→Image mapping** cập nhật visual prompts từ text
3. **Image→Text mapping** cập nhật text prompts từ visual  
4. **Các prompts được nhúng vào** text encoder và visual encoder ở các tầng khác nhau
5. **Mỗi tầng** thay prompts bằng phiên bản mới được học trong CrossModalPromptLearner
6. **Cuối cùng** cả 2 modality đều được mã hóa với learnable prompts đã được trao đổi thông tin

Điều này tạo ra một mô hình CLIP được cải tiến có thể học prompt chuyên biệt hóa dựa trên sự hiểu biết lẫn nhau giữa text và vision modality.
