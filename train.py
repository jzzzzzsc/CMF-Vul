import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CodeDataset(Dataset):
    def __init__(
        self,
        anchor_image_dir,
        anchor_graph_dir,
        anchor_source_dir,
        pos_image_dir,
        pos_graph_dir,
        pos_source_dir,
        processor,
        tokenizer,
        max_samples=None,
        image_ext=".png",
        source_ext=".c",
        graph_ext=".pkl",
    ):
        self.anchor_image_dir = anchor_image_dir
        self.anchor_graph_dir = anchor_graph_dir
        self.anchor_source_dir = anchor_source_dir
        self.pos_image_dir = pos_image_dir
        self.pos_graph_dir = pos_graph_dir
        self.pos_source_dir = pos_source_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.image_ext = image_ext
        self.source_ext = source_ext
        self.graph_ext = graph_ext

        self.filenames = []
        for img_name in os.listdir(anchor_image_dir):
            if self.max_samples and len(self.filenames) >= self.max_samples:
                break
            if not img_name.endswith(self.image_ext):
                continue

            base_name = img_name[: -len(self.image_ext)]
            try:
                int(img_name.split("_")[-1].split(".")[0])
            except Exception:
                continue

            a_pkl = os.path.join(anchor_graph_dir, f"{base_name}{self.graph_ext}")
            a_src = os.path.join(anchor_source_dir, f"{base_name}{self.source_ext}")

            p_img = os.path.join(pos_image_dir, img_name)
            p_pkl = os.path.join(pos_graph_dir, f"{base_name}{self.graph_ext}")
            p_src = os.path.join(pos_source_dir, f"{base_name}{self.source_ext}")

            if all(os.path.exists(p) for p in [a_pkl, a_src, p_img, p_pkl, p_src]):
                self.filenames.append(img_name)

    def __len__(self):
        return len(self.filenames)

    def _load_one(self, image_dir, graph_dir, source_dir, img_name):
        base_name = img_name[: -len(self.image_ext)]

        image = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        img_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        with open(os.path.join(graph_dir, f"{base_name}{self.graph_ext}"), "rb") as f:
            graph_data = pickle.load(f)[0]

        with open(os.path.join(source_dir, f"{base_name}{self.source_ext}"), "r", encoding="utf-8", errors="ignore") as f:
            source_code = f.read()

        source_inputs = self.tokenizer(
            source_code,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return img_tensor, graph_data, source_inputs

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        label = int(img_name.split("_")[-1].split(".")[0])

        a_img, a_graph, a_src = self._load_one(
            self.anchor_image_dir, self.anchor_graph_dir, self.anchor_source_dir, img_name
        )
        p_img, p_graph, p_src = self._load_one(
            self.pos_image_dir, self.pos_graph_dir, self.pos_source_dir, img_name
        )

        return a_img, a_graph, a_src, p_img, p_graph, p_src, label


def collate_fn(batch):
    a_images = [b[0] for b in batch]
    a_graphs = [b[1] for b in batch]
    a_sources = [b[2] for b in batch]
    p_images = [b[3] for b in batch]
    p_graphs = [b[4] for b in batch]
    p_sources = [b[5] for b in batch]
    labels = [b[6] for b in batch]

    def pack_sources(sources):
        return {
            "input_ids": torch.stack([x["input_ids"][0] for x in sources]),
            "attention_mask": torch.stack([x["attention_mask"][0] for x in sources]),
        }

    return (
        torch.stack(a_images),
        Batch.from_data_list(a_graphs),
        pack_sources(a_sources),
        torch.stack(p_images),
        Batch.from_data_list(p_graphs),
        pack_sources(p_sources),
        torch.tensor(labels, dtype=torch.long),
    )


class CodeEncoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.base_encoder = AutoModel.from_pretrained(model_path)
        self.feature_enhancer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=3072),
            num_layers=3,
        )
        self.feature_gate = nn.Sequential(nn.Linear(768, 768), nn.Sigmoid())

    def forward(self, input_ids, attention_mask):
        base_features = self.base_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        enhanced = self.feature_enhancer(base_features)
        gate = self.feature_gate(base_features)
        return (enhanced * gate).mean(dim=1)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=768):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class MultimodalFusion(nn.Module):
    def __init__(self, img_dim=768, graph_dim=768, code_dim=768, proj_dim=256):
        super().__init__()
        self.proj_img = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
        self.proj_graph = nn.Sequential(nn.Linear(graph_dim, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
        self.proj_code = nn.Sequential(nn.Linear(code_dim, proj_dim), nn.GELU(), nn.LayerNorm(proj_dim))
        self.fusion_weights = nn.Sequential(
            nn.Linear(proj_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, img_feat, graph_feat, code_feat):
        v = self.proj_img(img_feat)
        g = self.proj_graph(graph_feat)
        c = self.proj_code(code_feat)
        combined = torch.cat([v, g, c], dim=1)
        weights = self.fusion_weights(combined)
        fused = weights[:, 0:1] * v + weights[:, 1:2] * g + weights[:, 2:3] * c
        return fused, weights


class CLIPAlignWithCode(nn.Module):
    def __init__(self, num_classes=2, fused_dim=256, proj_dim=256):
        super().__init__()
        self.fusion = MultimodalFusion(proj_dim=fused_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.fused_proj = nn.Sequential(nn.Linear(fused_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def encode_fused(self, img_feat, graph_feat, code_feat):
        fused_feat, weights = self.fusion(img_feat, graph_feat, code_feat)
        z = F.normalize(self.fused_proj(fused_feat), dim=-1)
        return fused_feat, z, weights

    def forward(self, a_img_feat, a_graph_feat, a_code_feat, p_img_feat, p_graph_feat, p_code_feat):
        a_fused, a_z, a_w = self.encode_fused(a_img_feat, a_graph_feat, a_code_feat)
        _, p_z, _ = self.encode_fused(p_img_feat, p_graph_feat, p_code_feat)

        a_img_norm = F.normalize(self.fusion.proj_img(a_img_feat), dim=-1)
        a_graph_norm = F.normalize(self.fusion.proj_graph(a_graph_feat), dim=-1)
        a_code_norm = F.normalize(self.fusion.proj_code(a_code_feat), dim=-1)

        logits = self.classifier(a_fused)

        return {
            "logits": logits,
            "anchor_z": a_z,
            "pos_z": p_z,
            "contrastive_features": (a_img_norm, a_graph_norm, a_code_norm),
            "fusion_weights": a_w,
            "logit_scale": self.logit_scale.exp(),
        }


def pool_graph(node_emb, batch_obj):
    return torch.stack([node_emb[batch_obj.batch == i].mean(0) for i in range(batch_obj.num_graphs)])


def compute_lmc(outputs, device):
    img_feat, graph_feat, code_feat = outputs["contrastive_features"]
    scale = outputs["logit_scale"]

    logits_img_graph = scale * (img_feat @ graph_feat.t())
    logits_graph_img = logits_img_graph.t()

    logits_img_code = scale * (img_feat @ code_feat.t())
    logits_code_img = logits_img_code.t()

    bsz = img_feat.size(0)
    targets = torch.arange(bsz, device=device)

    return (
        F.cross_entropy(logits_img_graph, targets)
        + F.cross_entropy(logits_graph_img, targets)
        + F.cross_entropy(logits_img_code, targets)
        + F.cross_entropy(logits_code_img, targets)
    ) / 4.0


def adaptive_hard_negative_contrastive_loss(anchor_z, pos_z, labels, tau=0.07, eps=1e-8):
    anchor_z = F.normalize(anchor_z, dim=-1)
    pos_z = F.normalize(pos_z, dim=-1)

    sim = (anchor_z @ pos_z.t()) / tau
    exp_sim = torch.exp(sim)
    B = labels.size(0)

    label_ne = labels.view(-1, 1) != labels.view(1, -1)
    not_self = ~torch.eye(B, dtype=torch.bool, device=labels.device)
    neg_mask = label_ne & not_self

    pos_exp = exp_sim.diag()
    sum_neg = (exp_sim * neg_mask).sum(dim=1)
    n_neg = neg_mask.sum(dim=1)

    denom_avg = (sum_neg.view(B, 1) - exp_sim) / (n_neg.view(B, 1) - 1).clamp(min=1).to(exp_sim.dtype)
    S = exp_sim / (exp_sim + denom_avg + eps)
    S = S * neg_mask.to(exp_sim.dtype)

    weighted_neg = (S * exp_sim).sum(dim=1)
    denom = pos_exp + weighted_neg + eps
    return (-torch.log(pos_exp / denom)).mean()


def joint_loss(outputs, labels, device, alpha=0.33, beta=0.34, gamma=0.33, tau=0.07):
    lce = F.cross_entropy(outputs["logits"], labels)
    lmc = compute_lmc(outputs, device)
    lcl = adaptive_hard_negative_contrastive_loss(outputs["anchor_z"], outputs["pos_z"], labels, tau=tau)
    total = alpha * lmc + beta * lce + gamma * lcl
    return total


def train_epoch(model, gcn, code_encoder, vision_model, loader, optimizer, device, alpha=0.33, beta=0.34, gamma=0.33, tau=0.07):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        a_images, a_graphs, a_sources, p_images, p_graphs, p_sources, labels = batch
        a_images = a_images.to(device)
        p_images = p_images.to(device)
        a_graphs = a_graphs.to(device)
        p_graphs = p_graphs.to(device)
        a_sources = {k: v.to(device) for k, v in a_sources.items()}
        p_sources = {k: v.to(device) for k, v in p_sources.items()}
        labels = labels.to(device)

        with torch.no_grad():
            a_img_feat = vision_model(a_images).last_hidden_state.mean(dim=1)
            p_img_feat = vision_model(p_images).last_hidden_state.mean(dim=1)

        a_graph_feat = pool_graph(gcn(a_graphs), a_graphs)
        p_graph_feat = pool_graph(gcn(p_graphs), p_graphs)

        a_code_feat = code_encoder(**a_sources)
        p_code_feat = code_encoder(**p_sources)

        outputs = model(a_img_feat, a_graph_feat, a_code_feat, p_img_feat, p_graph_feat, p_code_feat)
        loss = joint_loss(outputs, labels, device, alpha=alpha, beta=beta, gamma=gamma, tau=tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs["logits"], dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return {"loss": total_loss / max(len(loader), 1), "accuracy": accuracy_score(all_labels, all_preds)}


def validate(model, gcn, code_encoder, vision_model, loader, device, alpha=0.33, beta=0.34, gamma=0.33, tau=0.07):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            a_images, a_graphs, a_sources, p_images, p_graphs, p_sources, labels = batch
            a_images = a_images.to(device)
            p_images = p_images.to(device)
            a_graphs = a_graphs.to(device)
            p_graphs = p_graphs.to(device)
            a_sources = {k: v.to(device) for k, v in a_sources.items()}
            p_sources = {k: v.to(device) for k, v in p_sources.items()}
            labels = labels.to(device)

            a_img_feat = vision_model(a_images).last_hidden_state.mean(dim=1)
            p_img_feat = vision_model(p_images).last_hidden_state.mean(dim=1)

            a_graph_feat = pool_graph(gcn(a_graphs), a_graphs)
            p_graph_feat = pool_graph(gcn(p_graphs), p_graphs)

            a_code_feat = code_encoder(**a_sources)
            p_code_feat = code_encoder(**p_sources)

            outputs = model(a_img_feat, a_graph_feat, a_code_feat, p_img_feat, p_graph_feat, p_code_feat)
            loss = joint_loss(outputs, labels, device, alpha=alpha, beta=beta, gamma=gamma, tau=tau)

            total_loss += loss.item()
            preds = torch.argmax(outputs["logits"], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {
        "anchor_image_dir": "",
        "anchor_graph_dir": "",
        "anchor_source_dir": "",
        "pos_image_dir": "",
        "pos_graph_dir": "",
        "pos_source_dir": "",
        "vision_model_path": "",
        "code_model_path": "",
        "batch_size": 32,
        "num_workers": 8,
        "epochs": 50,
        "seed": 42,
        "alpha": 0.33,
        "beta": 0.34,
        "gamma": 0.33,
        "tau": 0.07,
        "lr_model": 1e-4,
        "lr_gcn": 1e-3,
        "lr_code": 5e-5,
        "weight_decay": 0.01,
        "max_samples": None,
    }

    processor = AutoImageProcessor.from_pretrained(cfg["vision_model_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["code_model_path"])

    dataset = CodeDataset(
        anchor_image_dir=cfg["anchor_image_dir"],
        anchor_graph_dir=cfg["anchor_graph_dir"],
        anchor_source_dir=cfg["anchor_source_dir"],
        pos_image_dir=cfg["pos_image_dir"],
        pos_graph_dir=cfg["pos_graph_dir"],
        pos_source_dir=cfg["pos_source_dir"],
        processor=processor,
        tokenizer=tokenizer,
        max_samples=cfg["max_samples"],
    )

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(cfg["seed"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
    )

    vision_model = AutoModel.from_pretrained(cfg["vision_model_path"]).to(device).eval()
    code_encoder = CodeEncoder(cfg["code_model_path"]).to(device)
    gcn = GCN(in_dim=dataset[0][1].x.size(1)).to(device)
    model = CLIPAlignWithCode(num_classes=2).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg["lr_model"]},
            {"params": gcn.parameters(), "lr": cfg["lr_gcn"]},
            {"params": code_encoder.parameters(), "lr": cfg["lr_code"]},
        ],
        weight_decay=cfg["weight_decay"],
    )

    best_f1 = -1.0
    for epoch in range(cfg["epochs"]):
        train_metrics = train_epoch(
            model, gcn, code_encoder, vision_model, train_loader, optimizer, device,
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], tau=cfg["tau"]
        )
        val_metrics = validate(
            model, gcn, code_encoder, vision_model, val_loader, device,
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], tau=cfg["tau"]
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {"model": model.state_dict(), "gcn": gcn.state_dict(), "code_encoder": code_encoder.state_dict()},
                "best_model.pth",
            )

        print(
            f"Epoch {epoch+1}/{cfg['epochs']} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_recall={val_metrics['recall']:.4f} val_precision={val_metrics['precision']:.4f}"
        )

    ckpt = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    gcn.load_state_dict(ckpt["gcn"])
    code_encoder.load_state_dict(ckpt["code_encoder"])

    test_metrics = validate(
        model, gcn, code_encoder, vision_model, test_loader, device,
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], tau=cfg["tau"]
    )
    print(
        f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} test_recall={test_metrics['recall']:.4f} test_precision={test_metrics['precision']:.4f}"
    )


if __name__ == "__main__":
    main()
