
import os
import glob
import pickle
import networkx as nx
import torch
import torch_geometric

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data
from tqdm import tqdm


dir_name = ""
out_path = ""
model_path = ""

device = torch.device(
    "cuda:3" if torch.cuda.is_available() and torch.cuda.device_count() > 3
    else "cuda:0" if torch.cuda.is_available()
    else "cpu"
)

tokenizer = None
bertmodel = None


def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def extract_code(raw_label):
    code = str(raw_label).strip()

    if code.startswith('"') and code.endswith('"'):
        code = code[1:-1]

    if "," in code:
        code = code.split(",", 1)[1]

    code = code.split("\\n")[0]
    code = code.split("\n")[0]
    code = code.replace("static void", "void")

    return code.strip()


def sentence_embedding(code):

    with torch.no_grad():

        nl_tokens = tokenizer.tokenize(code)

        max_length = 512 - 2

        if len(nl_tokens) > max_length:
            nl_tokens = nl_tokens[:max_length]

        tokens_ids = tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + nl_tokens + ['[SEP]']
        )

        tokens_ids = torch.tensor(
            tokens_ids, dtype=torch.long
        ).to(device)

        context_embeddings = bertmodel(
            tokens_ids[None, :]
        )[0]

        context_embeddings = torch.mean(
            context_embeddings, dim=1
        )

        line_vec = context_embeddings.squeeze(0)

        return line_vec.detach().cpu()


def image_generation(dot, dot_name):

    try:

        graph = graph_extraction(dot)

        node_list = list(graph.nodes())

        if len(node_list) == 0:
            return None

        node_mapping = {
            node: idx
            for idx, node in enumerate(node_list)
        }

        edges = [
            [node_mapping[u], node_mapping[v]]
            for u, v in graph.edges()
        ]

        if len(edges) > 0:
            edge_index = torch.tensor(
                edges, dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )

        nodes_emb = []

        for node in node_list:

            raw_label = graph.nodes[node].get(
                "label", ""
            )

            code = extract_code(raw_label)

            line_vec = sentence_embedding(code)

            nodes_emb.append(line_vec)

        x = torch.stack(nodes_emb, dim=0)

        if dot_name.endswith("0"):
            y = torch.tensor(0.0)
        else:
            y = torch.tensor(1.0)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        print(
            f"{dot_name}: "
            f"x={data.x.shape}, "
            f"edge_index={data.edge_index.shape}, "
            f"y={data.y.item()}"
        )

        return [data]

    except Exception as e:

        print(f"Error processing {dot_name}: {e}")

        return None


def write_to_pkl(dot, out):

    dot_name = os.path.splitext(
        os.path.basename(dot)
    )[0]

    out_file_path = os.path.join(
        out, f"{dot_name}.pkl"
    )

    if os.path.exists(out_file_path):
        return "skipped"

    dataset = image_generation(dot, dot_name)

    if dataset is None:
        return "failed"

    temp_path = out_file_path + ".tmp"

    try:

        with open(temp_path, "wb") as f:
            pickle.dump(dataset, f)

        os.replace(temp_path, out_file_path)

        return "success"

    except Exception as e:

        print(f"Error saving {dot_name}: {e}")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return "failed"


def main():

    global tokenizer, bertmodel

    if not dir_name or not out_path or not model_path:
        raise ValueError(
            "Please set dir_name, out_path and model_path."
        )

    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    bertmodel = AutoModel.from_pretrained(model_path)

    bertmodel.to(device)
    bertmodel.eval()

    dotfiles = sorted(
        glob.glob(
            os.path.join(dir_name, "*.dot")
        )
    )

    print("Device:", device)
    print("Total DOT files:", len(dotfiles))

    success_count = 0
    skip_count = 0
    fail_count = 0

    for dot in tqdm(dotfiles, desc="Processing DOT files"):

        result = write_to_pkl(dot, out_path)

        if result == "success":
            success_count += 1
        elif result == "skipped":
            skip_count += 1
        else:
            fail_count += 1

    print("Successfully processed:", success_count)
    print("Skipped:", skip_count)
    print("Failed:", fail_count)


if __name__ == "__main__":
    main()
