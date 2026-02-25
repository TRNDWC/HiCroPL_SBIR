"""
Unit tests for HiCroPL components.

Each test creates dummy data with explicit shapes, runs the component,
and verifies output shapes and gradient flow.

Run: python -m pytest tests/test_hicropl.py -v
  or: python -m tests.test_hicropl
"""

import torchx
import torch.nn as nn
import unittest


class TestAttentionPooling(unittest.TestCase):
    """Test Layer-specific Knowledge Proxy (LKP).
    
    Input:
        token_query:    [1, hidden_size] - learnable proxy token
        sequence_key:   [n_ctx, hidden_size] - prompt tokens
        sequence_value: [n_ctx, hidden_size] - prompt tokens
    Output:
        [1, hidden_size] - compressed proxy token
    """

    def setUp(self):
        from src.hicropl import AttentionPooling
        self.hidden_size = 512
        self.n_ctx = 4
        self.model = AttentionPooling(
            hidden_size=self.hidden_size, num_attention_heads=8
        )

    def test_output_shape(self):
        """Verify output shape is [1, hidden_size]."""
        query = torch.randn(1, self.hidden_size)
        key = torch.randn(self.n_ctx, self.hidden_size)
        value = torch.randn(self.n_ctx, self.hidden_size)

        output = self.model(query, key, value)
        self.assertEqual(output.shape, (1, self.hidden_size))

    def test_gradient_flow(self):
        """Verify gradients flow to query, key, value, and model parameters."""
        query = torch.randn(1, self.hidden_size, requires_grad=True)
        key = torch.randn(self.n_ctx, self.hidden_size, requires_grad=True)
        value = torch.randn(self.n_ctx, self.hidden_size, requires_grad=True)

        output = self.model(query, key, value)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        # Check model params have gradients
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")

    def test_visual_dim(self):
        """Verify LKP works with visual dimension (768)."""
        from src.hicropl import AttentionPooling
        model = AttentionPooling(hidden_size=768, num_attention_heads=8)
        query = torch.randn(1, 768)
        key = torch.randn(self.n_ctx, 768)
        output = model(query, key, key)
        self.assertEqual(output.shape, (1, 768))


class TestCrossPromptAttention(unittest.TestCase):
    """Test Multi-scale Knowledge Mapper.
    
    Input:
        q: [n_target, hidden_size] - target modality prompts
        k: [n_source, encoder_hidden_size] - source modality proxy tokens
        v: [n_source, encoder_hidden_size] - source modality proxy tokens
    Output:
        [n_target, hidden_size] - updated target prompts
    """

    def setUp(self):
        from src.hicropl import CrossPromptAttention
        self.v_dim = 768
        self.ctx_dim = 512
        self.n_ctx = 4
        self.cross_layer = 4

    def test_text_to_visual_shape(self):
        """T->V: target=visual(768), source=text(512). 
        Output should be [cross_layer*n_ctx, 768]."""
        from src.hicropl import CrossPromptAttention
        mapper = CrossPromptAttention(
            hidden_size=self.v_dim, 
            encoder_hidden_size=self.ctx_dim,
            num_attention_heads=8,
        )
        n_target = self.cross_layer * self.n_ctx  # 16
        n_source = self.cross_layer               # 4

        q = torch.randn(n_target, self.v_dim)
        k = torch.randn(n_source, self.ctx_dim)

        output = mapper(q, k, k)
        self.assertEqual(output.shape, (n_target, self.v_dim))

    def test_visual_to_text_shape(self):
        """V->T: target=text(512), source=visual(768).
        Output should be [n_later*n_ctx, 512]."""
        from src.hicropl import CrossPromptAttention
        mapper = CrossPromptAttention(
            hidden_size=self.ctx_dim,
            encoder_hidden_size=self.v_dim,
            num_attention_heads=8,
        )
        n_later = 5
        n_target = n_later * self.n_ctx  # 20
        n_source = n_later               # 5

        q = torch.randn(n_target, self.ctx_dim)
        k = torch.randn(n_source, self.v_dim)

        output = mapper(q, k, k)
        self.assertEqual(output.shape, (n_target, self.ctx_dim))

    def test_gradient_flow(self):
        """Verify gradients flow through mapper to all parameters."""
        from src.hicropl import CrossPromptAttention
        mapper = CrossPromptAttention(
            hidden_size=self.v_dim,
            encoder_hidden_size=self.ctx_dim,
            num_attention_heads=8,
        )
        q = torch.randn(8, self.v_dim, requires_grad=True)
        k = torch.randn(4, self.ctx_dim, requires_grad=True)

        output = mapper(q, k, k)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        for name, p in mapper.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")


class TestTextEncoder(unittest.TestCase):
    """Test TextEncoder with deep prompt injection.
    
    Input:
        prompts: [n_cls, 77, ctx_dim] - first layer text input
        tokenized_prompts: [n_cls, 77] - tokenized (for EOT position)
        deep_prompts_text: list of L-1 tensors [n_ctx, ctx_dim]
    Output:
        [n_cls, embed_dim] - text features
    """

    def setUp(self):
        """Create a minimal mock CLIP text encoder."""
        self.ctx_dim = 512
        self.embed_dim = 512
        self.n_ctx = 4
        self.n_cls = 3
        self.n_layers = 4
        self.seq_len = 77

        # Build a minimal mock CLIP model
        class MockCLIP:
            dtype = torch.float32

        class MockTransformer(nn.Module):
            def __init__(self, width, layers, heads):
                super().__init__()
                class ResidualAttentionBlock(nn.Module):
                    def __init__(self, d_model):
                        super().__init__()
                        self.attn = nn.Linear(d_model, d_model)
                    def forward(self, x):
                        return x + self.attn(x)
                self.resblocks = nn.Sequential(
                    *[ResidualAttentionBlock(width) for _ in range(layers)]
                )

        self.transformer = MockTransformer(self.ctx_dim, self.n_layers, 8)

        mock_clip = MockCLIP()
        mock_clip.transformer = self.transformer
        mock_clip.positional_embedding = nn.Parameter(
            torch.randn(self.seq_len, self.ctx_dim)
        )
        mock_clip.ln_final = nn.LayerNorm(self.ctx_dim)
        mock_clip.text_projection = nn.Parameter(
            torch.randn(self.ctx_dim, self.embed_dim)
        )
        self.mock_clip = mock_clip

    def test_output_shape(self):
        """Verify output shape is [n_cls, embed_dim]."""
        from src.hicropl import TextEncoder
        encoder = TextEncoder(self.mock_clip)

        prompts = torch.randn(self.n_cls, self.seq_len, self.ctx_dim)
        # Simulate tokenized prompts with EOT at position 5+n_ctx
        tokenized = torch.zeros(self.n_cls, self.seq_len, dtype=torch.long)
        tokenized[:, 1 + self.n_ctx + 2] = 49407  # EOT token (highest value)

        deep_prompts = [
            torch.randn(self.n_ctx, self.ctx_dim) for _ in range(self.n_layers - 1)
        ]

        output = encoder(prompts, tokenized, deep_prompts)
        self.assertEqual(output.shape, (self.n_cls, self.embed_dim))

    def test_no_deep_prompts(self):
        """Verify encoder works with empty deep prompts (shallow mode)."""
        from src.hicropl import TextEncoder
        encoder = TextEncoder(self.mock_clip)

        prompts = torch.randn(self.n_cls, self.seq_len, self.ctx_dim)
        tokenized = torch.zeros(self.n_cls, self.seq_len, dtype=torch.long)
        tokenized[:, 5] = 49407

        output = encoder(prompts, tokenized, [])
        self.assertEqual(output.shape, (self.n_cls, self.embed_dim))


class TestVisualEncoder(unittest.TestCase):
    """Test VisualEncoder with deep prompt injection.
    
    Input:
        image: [B, 3, 224, 224] - input image
        first_visual_prompt: [n_ctx, v_dim] - first layer prompt
        deeper_visual_prompts: list of L-1 tensors [n_ctx, v_dim]
    Output:
        [B, embed_dim] - image features
    """

    def setUp(self):
        self.batch_size = 2
        self.v_dim = 768
        self.embed_dim = 512
        self.n_ctx = 4
        self.n_layers = 4
        self.patch_size = 32
        self.img_size = 224
        self.grid = self.img_size // self.patch_size  # 7

        class MockCLIP:
            dtype = torch.float32

        class MockViT(nn.Module):
            def __init__(self, width, layers, heads, patch_size, img_size, output_dim):
                super().__init__()
                class ResidualAttentionBlock(nn.Module):
                    def __init__(self, d_model):
                        super().__init__()
                        self.attn = nn.Linear(d_model, d_model)
                    def forward(self, x):
                        return x + self.attn(x)
                self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
                scale = width ** -0.5
                grid = img_size // patch_size
                self.class_embedding = nn.Parameter(scale * torch.randn(width))
                self.positional_embedding = nn.Parameter(scale * torch.randn(grid**2 + 1, width))
                self.ln_pre = nn.LayerNorm(width)
                self.transformer = nn.Module()
                self.transformer.resblocks = nn.Sequential(
                    *[ResidualAttentionBlock(width) for _ in range(layers)]
                )
                self.ln_post = nn.LayerNorm(width)
                self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        mock_clip = MockCLIP()
        mock_clip.visual = MockViT(
            self.v_dim, self.n_layers, 8, self.patch_size, self.img_size, self.embed_dim
        )
        self.mock_clip = mock_clip

    def test_output_shape(self):
        """Verify output shape is [B, embed_dim]."""
        from src.hicropl import VisualEncoder
        encoder = VisualEncoder(self.mock_clip)

        image = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        first_prompt = torch.randn(self.n_ctx, self.v_dim)
        deeper_prompts = [
            torch.randn(self.n_ctx, self.v_dim) for _ in range(self.n_layers - 1)
        ]

        output = encoder(image, first_prompt, deeper_prompts)
        self.assertEqual(output.shape, (self.batch_size, self.embed_dim))

    def test_no_deep_prompts(self):
        """Verify encoder works with only first-layer prompt (shallow mode)."""
        from src.hicropl import VisualEncoder
        encoder = VisualEncoder(self.mock_clip)

        image = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        first_prompt = torch.randn(self.n_ctx, self.v_dim)

        output = encoder(image, first_prompt, [])
        self.assertEqual(output.shape, (self.batch_size, self.embed_dim))


class TestCrossModalPromptLearner(unittest.TestCase):
    """Test CrossModalPromptLearner bidirectional flow.
    
    Input (forward):
        classnames: list of class name strings
    Output:
        text_input: [n_cls, 77, ctx_dim]
        tokenized_prompts: [n_cls, 77]
        first_visual_prompt: [n_ctx, v_dim]
        deeper_text_prompts: list of L-1 tensors [n_ctx, ctx_dim]
        deeper_visual_prompts: list of L-1 tensors [n_ctx, v_dim]
    """

    def _create_prompt_learner(self, n_ctx=4, prompt_depth=9, cross_layer=4):
        """Helper to create CrossModalPromptLearner with mock CLIP."""
        from src.hicropl import CrossModalPromptLearner
        class MockCLIP:
            dtype = torch.float32
            
        class MockTokenEmbedding:
            def __call__(self, x):
                return torch.randn(x.shape[0], x.shape[1], 512)
                
        class MockViT:
            class MockConv:
                weight = torch.randn(768, 3, 32, 32)
            conv1 = MockConv()
            
        class MockLN:
            weight = torch.randn(512)
            
        mock_clip = MockCLIP()
        mock_clip.token_embedding = MockTokenEmbedding()
        mock_clip.visual = MockViT()
        mock_clip.ln_final = MockLN()
        
        # Mock src.clip.clip to avoid torchvision import in __init__ and forward()
        import sys
        class MockModule:
            pass
        mock_clip_pkg = MockModule()
        mock_clip_module = MockModule()
        
        def mock_tokenize(texts):
            if isinstance(texts, str):
                return torch.zeros(1, 77, dtype=torch.long)
            return torch.zeros(len(texts), 77, dtype=torch.long)
            
        mock_clip_module.tokenize = mock_tokenize
        mock_clip_pkg.clip = mock_clip_module
        
        sys.modules['src.clip'] = mock_clip_pkg
        sys.modules['src.clip.clip'] = mock_clip_module
        
        learner = CrossModalPromptLearner(
            clip_model=mock_clip,
            n_ctx=n_ctx,
            prompt_depth=prompt_depth,
            cross_layer=cross_layer,
            ctx_init="a photo of a",
            use_fp16=False,
        )
        
        return learner

    def test_output_shapes(self):
        """Verify all output tensor shapes."""
        n_ctx = 4
        prompt_depth = 9
        cross_layer = 4
        learner = self._create_prompt_learner(n_ctx, prompt_depth, cross_layer)

        classnames = ["cat", "dog", "car"]
        text_input, tokenized, first_vis, deeper_text, deeper_vis = learner(classnames)

        n_cls = len(classnames)
        self.assertEqual(text_input.shape, (n_cls, 77, 512))
        self.assertEqual(tokenized.shape, (n_cls, 77))
        self.assertEqual(first_vis.shape, (n_ctx, 768))
        self.assertEqual(len(deeper_text), prompt_depth - 1)
        self.assertEqual(len(deeper_vis), prompt_depth - 1)
        for i, dt in enumerate(deeper_text):
            self.assertEqual(dt.shape, (n_ctx, 512), f"deeper_text[{i}] shape mismatch")
        for i, dv in enumerate(deeper_vis):
            self.assertEqual(dv.shape, (n_ctx, 768), f"deeper_vis[{i}] shape mismatch")

    def test_bidirectional_flow_modifies_prompts(self):
        """Verify that bidirectional flow changes visual and text prompt values."""
        learner = self._create_prompt_learner()

        # Snapshot prompt values before forward
        vis_before = [p.data.clone() for p in learner.cross_prompts_visual]
        text_before = [p.data.clone() for p in learner.cross_prompts_text]

        _ = learner(["cat", "dog"])

        # Early visual prompts (0..cross_layer-1) should be modified by T->V
        for i in range(learner.cross_layer):
            self.assertFalse(
                torch.equal(vis_before[i], learner.cross_prompts_visual[i].data),
                f"Visual prompt {i} was NOT modified by T->V flow",
            )

        # Later text prompts (cross_layer..depth-1) should be modified by V->T
        for i in range(learner.cross_layer, learner.prompt_depth):
            self.assertFalse(
                torch.equal(text_before[i], learner.cross_prompts_text[i].data),
                f"Text prompt {i} was NOT modified by V->T flow",
            )

    def test_parameter_count(self):
        """Verify the module has the expected learnable parameters."""
        learner = self._create_prompt_learner(n_ctx=4, prompt_depth=9, cross_layer=4)
        
        param_groups = {
            "cross_prompts_text": 0,
            "cross_prompts_visual": 0,
            "text2visual_net": 0,
            "visual2text_net": 0,
            "attn_pooling": 0,
            "proxy_tokens": 0,
        }
        total = 0
        for name, p in learner.named_parameters():
            total += p.numel()
            if "cross_prompts_text" in name:
                param_groups["cross_prompts_text"] += p.numel()
                print(f"Found text parameter: {name} | shape: {p.shape}")
            elif "cross_prompts_visual" in name:
                param_groups["cross_prompts_visual"] += p.numel()
            elif "text2visual" in name:
                param_groups["text2visual_net"] += p.numel()
            elif "visual2text" in name:
                param_groups["visual2text_net"] += p.numel()
            elif "attn_pooling" in name:
                param_groups["attn_pooling"] += p.numel()
            elif "proxy_token" in name:
                param_groups["proxy_tokens"] += p.numel()

        # Text prompts: 9 layers * 4 tokens * 512 dim = 18432
        self.assertEqual(param_groups["cross_prompts_text"], 9 * 4 * 512)
        # Visual prompts: 9 layers * 4 tokens * 768 dim = 27648
        self.assertEqual(param_groups["cross_prompts_visual"], 9 * 4 * 768)
        
        print(f"\nTotal learnable parameters: {total:,}")
        for k, v in param_groups.items():
            print(f"  {k}: {v:,}")


if __name__ == "__main__":
    unittest.main()
